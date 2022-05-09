// a simple OpenCL kernel which copies all pixels from A to B
kernel void img_old(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

// a very simple histogram implementation
kernel void hist_simple(global const uchar* A, global int* B) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&B[bin_index]);
	//Atomic operations deal with race conditions, but serialise the access to global memory, and are slow
}
// Cumulative histogram
//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void hist_cum(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)
		atomic_add(&B[i], A[id]);
}
// Look-up table (LUT), normalised cumulative histogram
kernel void hist_lut(global int* A, global int* B) {
	int id = get_global_id(0);
	B[id] = A[id] * (double)255 / A[255];
}

//a simple OpenCL kernel which copies all pixels from A to B
kernel void back_proj(global uchar* A, global int* LUT, global uchar* B) {
	int id = get_global_id(0);
	B[id] = LUT[A[id]];
}