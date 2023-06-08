class CUDAGraph(torch._C._CUDAGraph):
 def __new__(cls):
  return super(CUDAGraph, cls).__new__(cls)
 def capture_begin(self, pool=None):
  if pool is None:
   super().capture_begin()
  else:
   super().capture_begin(pool)
 def capture_end(self):
  super().capture_end()
 def replay(self):
  super().replay()
 def reset(self):
  super().reset()
 def pool(self):
  return super().pool()
 def enable_debug_mode(self):
  return super().enable_debug_mode()
 def debug_dump(self, debug_path):
  return super().debug_dump(debug_path)
