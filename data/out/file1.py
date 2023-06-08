import gc
import torch
from ._utils import _dummy_type
from torch.utils._pytree import tree_flatten as _tree_flatten
from torch.utils._pytree import tree_unflatten as _tree_unflatten
if not hasattr(torch._C, '_CudaStreamBase'):
 torch._C.__dict__['_CUDAGraph'] = _dummy_type('_CUDAGraph')
 torch._C.__dict__['_graph_pool_handle'] = _dummy_type('_graph_pool_handle')
 torch._C.__dict__['_cuda_isCurrentStreamCapturing'] = _dummy_type('_cuda_isCurrentStreamCapturing')
from torch._C import _CUDAGraph  
from torch._C import _graph_pool_handle
from torch._C import _cuda_isCurrentStreamCapturing
