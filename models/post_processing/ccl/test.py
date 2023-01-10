import torch
import ccl_cuda
import time



img_3d = (torch.rand(128, 256, 256) * 255.0).to(torch.uint8).cuda()
print(img_3d)
img_2d = img_3d[0]
print(img_3d.shape)

print("ccl on gpu (12800 times) - single")
for i in range(5120):
    output = ccl_cuda.ccl(img_2d)

start_time = time.time()
for i in range(12800):
    output = ccl_cuda.ccl(img_2d)
    numbers = torch.unique(output)
print("using time: %.2fs" % (time.time() - start_time))
print(numbers)


print("ccl on gpu (12800 times) - batch")
for i in range(10):
    output = ccl_cuda.ccl_batch(img_3d)
    numbers = torch.unique(output.view(128, -1), dim=1)

start_time = time.time()
for i in range(100):
    output = ccl_cuda.ccl_batch(img_3d)
    numbers = torch.unique(output)
print(numbers)


print("using time: %.2fs" % (time.time() - start_time))