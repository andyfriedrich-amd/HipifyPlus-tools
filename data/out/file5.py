class graph:
 default_capture_stream = None
 def __init__(self, cuda_graph, pool=None, stream=None):
  if self.__class__.default_capture_stream is None:
   self.__class__.default_capture_stream = torch.cuda.Stream()
  self.pool = () if pool is None else (pool,)
  self.capture_stream = stream if stream is not None else self.__class__.default_capture_stream
  assert self.capture_stream is not None
  self.stream_ctx = torch.cuda.stream(self.capture_stream)
  self.cuda_graph = cuda_graph
 def __enter__(self):
  torch.cuda.synchronize()
  gc.collect()
  torch.cuda.empty_cache()
  self.stream_ctx.__enter__()
  self.cuda_graph.capture_begin(*self.pool)
 def __exit__(self, exc_type, exc_value, traceback):
  self.cuda_graph.capture_end()
  self.stream_ctx.__exit__(exc_type, exc_value, traceback)
