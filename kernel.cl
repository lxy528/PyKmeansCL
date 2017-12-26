typedef struct Point {
    float x, y;
}Point;


__kernel void assign (
				 __global float *centroids_x,
				 __global float *centroids_y,
				 __global float *data_x,
				 __global float *data_y,
				 __global int* partitioned,
				 __const int class_n,
				 __const int data_n,
				 __const float dbl_max)
{
	int data_i = get_global_id(0);
	Point t;
	float min_dist = dbl_max; 	
	int class_i;
	for(class_i = 0; class_i < class_n; class_i++){
			 t.x = data_x[data_i] - centroids_x[class_i];
			 t.y = data_y[data_i] - centroids_y[class_i];

			 float dist = t.x * t.x + t.y * t.y;
			 if (dist < min_dist) {
			 	partitioned[data_i] = class_i;
				min_dist = dist;
			 }
	}
	//printf("%d\n",partitioned[data_i]);
}

