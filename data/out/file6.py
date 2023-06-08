def make_graphed_callables(callables, sample_args, num_warmup_iters=3, allow_unused_input=False):
 if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
  raise RuntimeError("make_graphed_callables does not support the autocast caching. Please set `cache_enabled=False`.")
 just_one_callable = False
 if not isinstance(callables, tuple):
  just_one_callable = True
  callables = (callables,)
  sample_args = (sample_args,)
 flatten_sample_args = []
 for c, args in zip(callables, sample_args):
  if isinstance(c, torch.nn.Module):
   assert len(c._backward_hooks) == 0 and len(c._forward_hooks) == 0 and len(c._forward_pre_hooks) == 0, \
    "Modules must not have hooks registered at the time they are passed. However, registering hooks " + \
    "on modules after passing them through make_graphed_callables is allowed."
   assert all(b.requires_grad is False for b in c.buffers()), "In any :class:`~torch.nn.Module` passed to " + \
    ":func:`~make_graphed_callables`, only parameters may be trainable. All buffers must have " + \
    "``requires_grad=False``."
  flatten_arg, _ = _tree_flatten(args)
  flatten_sample_args.append(tuple(flatten_arg))
  assert all(isinstance(arg, torch.Tensor) for arg in flatten_arg), "In the beta API, sample_args " + \
   "for each callable must contain only Tensors. Other types are not allowed."
 per_callable_len_user_args = [len(args) for args in flatten_sample_args]
 per_callable_module_params = [tuple(c.parameters()) if isinstance(c, torch.nn.Module) else () for c in callables]
 per_callable_static_input_surfaces = [flatten_sample_args[i] + per_callable_module_params[i]
            for i in range(len(callables))]
 fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
 bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
 mempool = graph_pool_handle()
 torch.cuda.synchronize()
 with torch.cuda.stream(torch.cuda.Stream()):
  for func, args, static_input_surface in zip(callables, sample_args, per_callable_static_input_surfaces):
   for _ in range(num_warmup_iters):
    outputs, _ = _tree_flatten(func(*args))
    grad_inputs = torch.autograd.grad(outputs=tuple(o for o in outputs if o.requires_grad), inputs=tuple(i for i in static_input_surface if i.requires_grad), grad_outputs=tuple(torch.empty_like(o) for o in outputs if o.requires_grad), only_inputs=True, allow_unused=allow_unused_input)
   del outputs, grad_inputs
 torch.cuda.synchronize()
 per_callable_static_outputs = []
 per_callable_output_unflatten_spec = []
 for func, args, fwd_graph in zip(callables, sample_args, fwd_graphs):
  with torch.cuda.graph(fwd_graph, pool=mempool):
   outputs = func(*args)
  flatten_outputs, spec = _tree_flatten(outputs)
  per_callable_static_outputs.append(tuple(flatten_outputs))
  per_callable_output_unflatten_spec.append(spec)
 per_callable_static_grad_outputs = []
 per_callable_static_grad_inputs = []
 for static_input_surface, static_outputs, bwd_graph, module_params in \
   zip(reversed(per_callable_static_input_surfaces), reversed(per_callable_static_outputs), reversed(bwd_graphs), reversed(per_callable_module_params)):
  static_grad_outputs = tuple(torch.empty_like(o) if o.requires_grad else None for o in static_outputs)
  with torch.cuda.graph(bwd_graph, pool=mempool):
   grad_inputs = torch.autograd.grad(outputs=tuple(o for o in static_outputs if o.requires_grad), inputs=tuple(i for i in static_input_surface if i.requires_grad), grad_outputs=tuple(o for o in static_grad_outputs if o is not None), only_inputs=True, allow_unused=allow_unused_input)
  static_grad_inputs = []
  grad_idx = 0
  for arg in static_input_surface:
   if arg.requires_grad:
    static_grad_inputs.append(grad_inputs[grad_idx])
    grad_idx += 1
   else:
    static_grad_inputs.append(None)  
  static_grad_inputs = tuple(static_grad_inputs)  
  per_callable_static_grad_outputs.append(static_grad_outputs)
  per_callable_static_grad_inputs.append(static_grad_inputs)
 per_callable_static_grad_outputs = list(reversed(per_callable_static_grad_outputs))
 per_callable_static_grad_inputs = list(reversed(per_callable_static_grad_inputs))
 def make_graphed_autograd_function(fwd_graph, bwd_graph, module_params, len_user_args, output_unflatten_spec, static_input_surface, static_outputs, static_grad_outputs, static_grad_inputs):
  class Graphed(torch.autograd.Function):
   @staticmethod
   def forward(ctx, *inputs):
    for i in range(len_user_args):
     if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
      static_input_surface[i].copy_(inputs[i])
    fwd_graph.replay()
    assert isinstance(static_outputs, tuple)
    return tuple(o.detach() for o in static_outputs)
   @staticmethod
   @torch.autograd.function.once_differentiable
   def backward(ctx, *grads):
    assert len(grads) == len(static_grad_outputs)
    for g, grad in zip(static_grad_outputs, grads):
     if g is not None:
      if g.data_ptr() != grad.data_ptr():
       g.copy_(grad)
    bwd_graph.replay()
    assert isinstance(static_grad_inputs, tuple)
    return tuple(b.detach() if b is not None else b for b in static_grad_inputs)
  def functionalized(*user_args):
   flatten_user_args, _ = _tree_flatten(user_args)
   out = Graphed.apply(*(tuple(flatten_user_args) + module_params))
   return _tree_unflatten(out, output_unflatten_spec)
  return functionalized
 ret = []
 for i, func in enumerate(callables):
  graphed = make_graphed_autograd_function(fwd_graphs[i], bwd_graphs[i], per_callable_module_params[i], per_callable_len_user_args[i], per_callable_output_unflatten_spec[i], per_callable_static_input_surfaces[i], per_callable_static_outputs[i], per_callable_static_grad_outputs[i], per_callable_static_grad_inputs[i])
  if isinstance(func, torch.nn.Module):
   def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):
    def new_fwd(*user_args):
     if func.training == graph_training_state:
      return graphed(*user_args)
     else:
      return orig_fwd(*user_args)
    return new_fwd
   func.forward = make_graphed_forward(func, func.training, graphed, func.forward)  
   ret.append(func)
  else:
   ret.append(graphed)
 if just_one_callable:
  return ret[0]
 return tuple(ret)
