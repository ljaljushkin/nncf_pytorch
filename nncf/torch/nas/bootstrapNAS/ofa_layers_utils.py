

def sub_filter_start_end(kernel_size, sub_kernel_size):
	center = kernel_size // 2
	dev = sub_kernel_size // 2
	start, end = center - dev, center + dev + 1
	assert end - start == sub_kernel_size
	return start, end
