#include <cmath>
#include <iostream>
#include <stdio.h>
#include <cstdio>
#include "mymodel.h"

using namespace std;



//This code defines a Point3d struct with three float fields representing x, y, and z coordinates.
struct Point3d {
  float x, y, z;
};

//To define a type alias named Point3dPtr that points to a Point3d object
typedef Point3d* Point3dPtr;

//It defines a Ray struct that contains two pointers to Point3d objects, representing the endpoints of a ray.
struct Ray {
  Point3dPtr a;
  Point3dPtr b;
};


/*
The assign_values function takes a float array of length 3 and returns a new Point3d object
with the values from the array assigned to its x, y, and z fields.
*/
Point3d assign_values(float a[3]) {
  Point3d point;
  point.x = a[0];
  point.y = a[1];
  point.z = a[2];

  return point;
}

/*
The following functions are for performing vector math on Point3d objects,
including subtracting, calculating the dot product and cross product of two vectors,
and normalizing a vector.
*/

void sub_vector(Point3d* p1, Point3d* p2, Point3d* result1) {
  result1->x = p1->x - p2->x;
  result1->y = p1->y - p2->y;
  result1->z = p1->z - p2->z;
}


float dot_product(Point3d* p1, Point3d* p2) {
  return p1->x * p2->x + p1->y * p2->y + p1->z * p2->z;
}

void cross_product(Point3d* p1, Point3d* p2, Point3d* result3) {
  result3->x = p1->y * p2->z - p1->z * p2->y;
  result3->y = p1->z * p2->x - p1->x * p2->z;
  result3->z = p1->x * p2->y - p1->y * p2->x;
}

/*
The normalize function takes a pointer to a Point3d object and 
returns a normalized version of that vector in a new Point3d object.
*/
void normalize(Point3dPtr v0, Point3dPtr vn0) {
  float mag = std::sqrt(v0->x * v0->x + v0->y * v0->y + v0->z * v0->z);
  vn0->x = v0->x / mag;
  vn0->y = v0->y / mag;
  vn0->z = v0->z / mag;
}

typedef float transform3d[4][4];


/*
This function takes in two 3D points p2 and p3, and an empty 4x4 matrix rotTMatrix. 
It calculates the transpose of the rotation matrix that would align the vector from p2 to p3 with the positive z-axis. 
It does this by first computing the normal n of the vector from p2 to p3, 
then finding the vector u that is orthogonal to n and the vector from p2 to p3, and finally computing the vector v 
that is orthogonal to both n and u. The resulting rotation matrix is then stored in rotTMatrix.
*/
void rotation_transpose(Point3d* p2, Point3d* p3, transform3d rotTMatrix)
{
	Point3d tempU, u, v, n;
	normalize(p2, &n);

	cross_product(p3, p2, &tempU);
	normalize(&tempU, &u);

	cross_product(&n, &u, &v);

	rotTMatrix[0][0] = u.x;
	rotTMatrix[0][1] = v.x;
	rotTMatrix[0][2] = n.x;
	rotTMatrix[0][3] = 0.0;

	rotTMatrix[1][0] = u.y;
	rotTMatrix[1][1] = v.y;
	rotTMatrix[1][2] = n.y;
	rotTMatrix[1][3] = 0.0;

	rotTMatrix[2][0] = u.z;
	rotTMatrix[2][1] = v.z;
	rotTMatrix[2][2] = n.z;
	rotTMatrix[2][3] = 0.0;

	rotTMatrix[3][0] = 0.0;
	rotTMatrix[3][1] = 0.0;
	rotTMatrix[3][2] = 0.0;
	rotTMatrix[3][3] = 1.0;

}


/*
This function takes a 3D vector representing the View Reference Point (vrp) and a 4x4 matrix trans_inverse_matrix, 
and sets trans_inverse_matrix to be the inverse translation matrix for the given vrp.
*/
void translation_inverse(float vrp[3], transform3d trans_inverse_matrix)
{

	trans_inverse_matrix[0][0] = 1.0;
	trans_inverse_matrix[0][1] = 0.0;
	trans_inverse_matrix[0][2] = 0.0;
	trans_inverse_matrix[0][3] = vrp[0];

	trans_inverse_matrix[1][0] = 0.0;
	trans_inverse_matrix[1][1] = 1.0;
	trans_inverse_matrix[1][2] = 0.0;
	trans_inverse_matrix[1][3] = vrp[1];

	trans_inverse_matrix[2][0] = 0.0;
	trans_inverse_matrix[2][1] = 0.0;
	trans_inverse_matrix[2][2] = 1.0;
	trans_inverse_matrix[2][3] = vrp[2];

	trans_inverse_matrix[3][0] = 0.0;
	trans_inverse_matrix[3][1] = 0.0;
	trans_inverse_matrix[3][2] = 0.0;
	trans_inverse_matrix[3][3] = 1.0;
}


/*
The following function is to multiply two 4x4 matrices, transform1 and transform2, and store the result in result_transform. 
It first initializes a temporary 4x4 matrix result to 0, and then loops over the rows and columns of transform1 and transform2, 
respectively, and computes the dot product of the row of transform1 and the column of transform2 to populate the corresponding entry 
in result. Finally, it copies the values from result to result_transform.
*/
void mult_transforms(const float transform1[4][4], const float transform2[4][4], float result_transform[4][4]) {
  double result[4][4];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      result[i][j] = 0;
      for (int k = 0; k < 4; ++k) {
        result[i][j] += transform1[i][k] * transform2[k][j];
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      result_transform[i][j] = result[i][j];
    }
  }
}


/*
The following function maps a given integer value a in the range of 0 to CorR-1 to a corresponding floating-point value 
in the range of min to max. The function first computes the range difference temp1 between max and min. 
It then calculates the value of temp2 by multiplying the range difference by a and dividing the result by CorR-1. 
Finally, the function adds temp2 to max to obtain the mapped result, which is returned. 
The purpose of this function is likely to conumber_of_vert a given integer index to a floating-point value that represents 
a specific quantity within a certain range.
*/
float map(float max, float min, int a, int CorR) {
  float temp1, temp2, result;
  temp1 = min - max;
  temp2 = (temp1 * a) / (CorR - 1); //(map from (-0.0175~0.0175) to (0~511)
  result = max + temp2;
  return result;
}



/*
This function constructs a ray in 3D space given an image pixel location and camera parameters. 
The input parameters include the pixel location (i,j), focal length, image boundaries (xmin, xmax, ymin, ymax), 
a camera transformation matrix Mcw, and a viewpoint in world coordinates (vrp). 
The output is a Ray object containing the origin point (ray->a) and normalized direction vector (ray->b). 
The function first maps the pixel location to a 3D point in camera coordinates using the camera transformation matrix 
and focal length, and then maps that point to world coordinates using the viewpoint. 
The direction vector is computed as the difference between the 3D point and viewpoint, and then normalized. 
The resulting origin and direction are stored in the Ray object.
*/
void ray_construction(int i, int j, float focal, float xmin, float xmax, float ymin, float ymax, float Mcw[4][4], Point3dPtr vrp, Ray *ray) {
  Point3dPtr  p0, p1, v0, vn0;
  float xc, yc, f;

  p0 = new Point3d();
  p1 = new Point3d();
  v0 = new Point3d();
  vn0 = new Point3d();

  xc = map(xmax, xmin, j, COLS);
  yc = map(ymax, ymin, i, ROWS);
  f = focal;

  // vrp is already a 3d Point in the world coordinates , so it doesn't need transform.
  p0->x = vrp->x;
  p0->y = vrp->y;
  p0->z = vrp->z;

  p1->x = Mcw[0][0] * xc + Mcw[0][1] * yc + Mcw[0][2] * f + Mcw[0][3];
  p1->y = Mcw[1][0] * xc + Mcw[1][1] * yc + Mcw[1][2] * f + Mcw[1][3];
  p1->z = Mcw[2][0] * xc + Mcw[2][1] * yc + Mcw[2][2] * f + Mcw[2][3];

  // direction
  v0->x = p1->x - p0->x;
  v0->y = p1->y - p0->y;
  v0->z = p1->z - p0->z;
  normalize(v0, vn0);

  // finally the ray expression : R(t) = R0 + t * Rd , t > 0. Here is ray->a + t * (ray.b) ;
  ray->a = p0;
  ray->b = vn0;

}

// This function performs tri-linear interpolation to calculate the voxel value at a given point in 3D space
// Input arguments:
// - data: 3D array representing the volume dataset
// - cur_point: pointer to a point in 3D space where we want to calculate the voxel value
// Output:
// - val: interpolated voxel value at the given point
float tri_interpolation(unsigned char data[][ROWS][COLS], Point3dPtr cur_point) {
   // Calculate the indices of the eight neighboring voxels
  int i1 = floor(cur_point->x);
  int j1 = floor(cur_point->y);
  int k1 = floor(cur_point->z);
  int i2 = i1 + 1;
  int j2 = j1 + 1;
  int k2 = k1 + 1;

  // Calculate the fractional distance between the point and the eight neighboring voxels
  float dx1 = cur_point->x - i1;
  float dy1 = cur_point->y - j1;
  float dz1 = cur_point->z - k1;
  float dx2 = 1 - dx1;
  float dy2 = 1 - dy1;
  float dz2 = 1 - dz1;

  // Perform trilinear interpolation
  float val =
      dx2 * dy2 * dz2 * data[k1][j1][i1] +  // contribution from voxel at (i1, j1, k1)
      dx1 * dy2 * dz2 * data[k1][j1][i2] +  // contribution from voxel at (i2, j1, k1)
      dx2 * dy1 * dz2 * data[k1][j2][i1] +  // contribution from voxel at (i1, j2, k1)
      dx2 * dy2 * dz1 * data[k2][j1][i1] +  // contribution from voxel at (i1, j1, k2)
      dx1 * dy1 * dz2 * data[k1][j2][i2] +  // contribution from voxel at (i2, j2, k1)
      dx1 * dy2 * dz1 * data[k2][j1][i2] +  // contribution from voxel at (i2, j1, k2)
      dx2 * dy1 * dz1 * data[k2][j2][i1] +  // contribution from voxel at (i1, j2, k2)
      dx1 * dy1 * dz1 * data[k2][j2][i2];   // contribution from voxel at (i2, j2, k2)

  return val;
}      

/*

Computes shading for a given light ray position and a computed CT volume
Parameters:
lrp[3]: an array representing the position of the light ray
ct[SLCS][ROWS][COLS]: a 3D array representing the computed CT volume
ip: intensity of the light source
shading[SLCS][ROWS][COLS]: a 3D array representing the computed shading
Returns: void

This function computes shading for a given light ray position and a computed CT volume. 
It loops through all voxels in the CT volume and computes the length of the normal vector at each voxel. 
If the length of the normal vector is below a threshold value, the shading is set to 0. 
Otherwise, the function computes the normalized normal vector and light ray vector at the voxel, computes the dot product between them, 
scales the result by the intensity of the light source, and casts the result to an unsigned char. The resulting shading values are stored in a 3D array.
*/
                  
void compute_shading_volume(float lrp[3], unsigned char ct[SLCS][ROWS][COLS], float ip, unsigned char shading[SLCS][ROWS][COLS])
{
  // Set threshold value for normal vector length
  float threshold = 0.01f;
  // Loop through all voxels in the CT volume
  for (int k = 0; k < SLCS; ++k) {
    float lk = lrp[2] - k;
    for (int j = 0; j < ROWS; ++j) {
      float lj = lrp[1] - j;
      for (int i = 0; i < COLS; ++i) {
        float li = lrp[0] - i;
        // Compute length of the normal vector at this voxel
        float n_length = std::sqrt((ct[k][j][i+1] - ct[k][j][i]) * (ct[k][j][i+1] - ct[k][j][i])
                                     + (ct[k][j+1][i] - ct[k][j][i]) * (ct[k][j+1][i] - ct[k][j][i])
                                     + (ct[k+1][j][i] - ct[k][j][i]) * (ct[k+1][j][i] - ct[k][j][i]));
        // If the length of the normal vector is below the threshold value, set shading to 0
        if (n_length < threshold) {
          shading[k][j][i] = 0;
        } else {
          // Compute the normalized normal vector and light ray vector
          Point3d ln = {li, lj, lk};
          normalize(&ln, &ln);
          Point3d nn = {ct[k][j][i+1] - ct[k][j][i], ct[k][j+1][i] - ct[k][j][i], ct[k+1][j][i] - ct[k][j][i]};
          Point3d nn_norm;
          normalize(&nn, &nn_norm);
          // Compute the dot product between the normalized normal vector and light ray vector
          float temp = dot_product(&nn_norm, &ln);

          // Scale the result by the intensity of the light source and cast to unsigned char
          shading[k][j][i] = static_cast<unsigned char>(ip * temp);
        }
      }
    }
  }
}


// This function performs ray tracing through a CT volume to generate an image
// of the volume along the specified ray.
// Inputs:
// - ray: the ray to sample along
// - ts: an array of two floats representing the start and end points of the ray
// - ct: the CT volume to sample from
// - shading: the shading information to apply to the volume
// Returns:
// - The generated image as an unsigned char value

unsigned char volume_ray_tracing(Ray ray, float ts[2], unsigned char ct[SLCS][ROWS][COLS], unsigned char shading[SLCS][ROWS][COLS])
{
  // Initialize variables
  float Dt = 1.0; // Step size
  float C = 0.0; // Accumulating the color value
  float T = 1.0; // Accumulating the transparency
  float Alpha, Ci; // Attenuation and shading values for current sample
  float t0, t1, temp, t; // Start and end points of ray, loop variable
  unsigned char img = 0; // Generated image

  // Retrieve start and end points of ray
  t0 = ts[0];
  t1 = ts[1];

  // Swap t0 and t1 if needed
  if (t0 > t1)
  {
    temp = t0;
    t0 = t1;
    t1 = temp;
  }

  // Pre-compute ray direction multiplied by step size
  float dx = ray.b->x * Dt;
  float dy = ray.b->y * Dt;
  float dz = ray.b->z * Dt;

  // Pre-compute starting point
  float x = ray.a->x + dx * (t0 / Dt);
  float y = ray.a->y + dy * (t0 / Dt);
  float z = ray.a->z + dz * (t0 / Dt);

  // Loop through sampling points along the ray
  for (t = t0; t <= t1; t += Dt)
  {
    // Convert float position to integer voxel indices
    int i = (int)round(x);
    int j = (int)round(y);
    int k = (int)round(z);

    // Check if voxel is inside the CT volume
    if (i >= 0 && i < COLS && j >= 0 && j < ROWS && k >= 0 && k < SLCS)
    {
      // Calculate transparency and shading values
      Alpha = ct[k][j][i] / 255.f;
      Ci = shading[k][j][i];
      if(T>0.05){ //To ignore ones with very low transparency.
      T *= (1.0 - Alpha);
      C += T*Ci*Alpha ; // Accumulate color values
      
    }
    }

    // Move to next sampling point
    x += dx;
    y += dy;
    z += dz;
  }
  // Convert accumulated values to image value
  img = (unsigned char)(int)C;
  return img;
}



// This function determines the intersection of a given ray with the bounding box of the volume. The bounding box is defined by the dimensions of the volume (ROWS, COLS, SLCS), where ROWS represents the number of rows, COLS represents the number of columns, and SLCS represents the number of slices.
// The function takes in a pointer to a Ray struct, which contains the starting point of the ray (a) and its direction (b), and an array of floats ts[2], which will be populated with the intersection times of the ray with the bounding box.
// The function returns the number of intersection points found, which will be either 0, 1, or 2.
// The function first calculates the intersection times between the ray and each of the six bounding planes that define the box.
// If an intersection point lies within the bounds of the volume (i.e. within the range of valid indices for each dimension), its intersection time is added to the ts array.
// Finally, the function returns the number of intersection points found.
int ray_box_intersection(Ray *ray, float ts[2])
{
      int n = 0; // initialize n to zero
      float x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3; // initialize variables for x, y, and z
      float t0, t1, t2, t3, t4, t5;
    
    // calculate t value for when ray meets x=0 plane (for the left face of the bounding box).
    t0 = -1 * ray->a->x / ray->b->x;
    y0 = ray->a->y + t0 * ray->b->y;
    z0 = ray->a->z + t0 * ray->b->z;
    
    // calculate t value for when ray meets x=127 plane (for the right face of the bounding box).
    t1 = (127 - ray->a->x) / ray->b->x;
    y1 = ray->a->y + t1 * ray->b->y;
    z1 = ray->a->z + t1 * ray->b->z;
    
    // calculate t value for when ray meets y=0 plane (for the bottom face of the bounding box).
    t2 = -1 * ray->a->y / ray->b->y;
    x0 = ray->a->x + t2 * ray->b->x;
    z2 = ray->a->z + t2 * ray->b->z;
    
    // calculate t value for when ray meets y=127 plane (for the top face of the bounding box).
    t3 = (127 - ray->a->y) / ray->b->y;
    x1 = ray->a->x + t3 * ray->b->x;
    z3 = ray->a->z + t3 * ray->b->z;
    
    // calculate t value for when ray meets z=0 plane (for the back face of the bounding box).
    t4 = -1 * ray->a->z / ray->b->z;
    x2 = ray->a->x + t4 * ray->b->x;
    y2 = ray->a->y + t4 * ray->b->y;
    
    // calculate t value for when ray meets z=127 plane (for the front face of the bounding box).
    t5 = (127 - ray->a->z) / ray->b->z;
    x3 = ray->a->x + t5 * ray->b->x;
    y3 = ray->a->y + t5 * ray->b->y;
    
    // Check if each intersection point is within the bounds of the 3D volume.
    // If so, add the intersection

    // check if ray meets x=0 plane within bounds of display volume
    if (y0 > 0 && y0 < ROWS && z0 > 0 && z0 < SLCS)
    {
        ts[n] = t0; // add t value to ts array at index n
        n += 1; // increment n by one
    }
    
    // repeat for x=127, y=0, y=127, z=0, and z=127 planes
    if (y1 > 0 && y1 < ROWS && z1 > 0 && z1 < SLCS)
    {
        ts[n] = t1;
        n += 1;
    }
    if (x0 > 0 && x0 < COLS && z2 > 0 && z2 < SLCS)
    {
        ts[n] = t2;
        n += 1;
    }
    if (x1 > 0 && x1 < COLS && z3 > 0 && z3 < SLCS)
    {
        ts[n] = t3;
        n += 1;
    }
    if (y2 > 0 && y2 < ROWS && x2 > 0 && x2 < COLS)
    {
        ts[n] = t4;
        n += 1;
    }
    if (y3 > 0 && y3 < ROWS && x3 > 0 && x3 < COLS)
    {
        ts[n] = t5;
        n += 1;
    }
    
    // return the value of n, which is the number of intersection points found
    return n;
}



// The main function performs volume rendering using CT scan data.
// It reads in the CT scan data from the file "smallHead.den",
// computes the shading volume for the data, and then renders
// an image of the data using volume ray tracing. The resulting
// image is saved to the file "outImage.raw".

int main(int argc, char* argv[])
{
 // Declare a few variables
    FILE *infid, *outfid;    // input and output file ids
    int n;
    int d, e, f;

    // Load the CT data into the array
    // Open the file "smallHead.den" for reading in binary mode
    if ((infid = fopen("smallHead.den", "rb")) == NULL)
    {
        // If the file could not be opened, display an error message and terminate the program
        cerr << "Open CT DATA File Error." << endl;
        exit(1);
    }

    // Loop through each slice of the CT data and read it into the CT array
    for (f = 0; f < SLCS; f++)
    {
        // Read the slice of CT data into the array
        n = fread(&CT[f][0][0], sizeof(char), ROWS * COLS, infid);

        // Check if the slice was fully read, if not display an error message and terminate the program
        if (n < ROWS * COLS * sizeof(char))
        {
            cerr << "Read CT data slice " << f << " error." << endl;
            exit(1);
        }
    }

    // Compute the shading volume for the CT data
    compute_shading_volume(Light, CT, Ip, SHADING);

    //-------------------------------------

    // Declare some more variables for the image rendering
    int intersections;
    float ts[2];// for storing the intersection points t0 and t1;
    transform3d TIN1, RT1;
    transform3d Mcw;
    Point3d Pvpn, Pvup, Pvrp;

    // Initialize the Pvpn, Pvup, and Pvrp points
    Pvpn = assign_values(VPN);
    Pvup = assign_values(VUP);
    Pvrp = assign_values(VRP);

    // Compute the inverse TIN1 transform of the view reference point
    translation_inverse(VRP, TIN1);

    // Compute the transpose RT1 transform for the Pvpn and Pvup points
    rotation_transpose(&Pvpn, &Pvup, RT1);

    // Multiply the TIN1 and RT1 transforms to get the Mcw transform
    mult_transforms(TIN1, RT1, Mcw);

    // Loop through each pixel in the output image
    for (d = 0; d < IMG_ROWS; d++)
        for (e = 0; e < IMG_COLS; e++)
        {
            // Construct a ray V, started from the CenterOfProjection and passing through the pixel(i,j);
            Ray V;
            V.a = new Point3d;
            V.b = new Point3d;
            ray_construction(d, e, focal, xmin, xmax, ymin, ymax, Mcw, &Pvrp, &V);// construct Ray V

            // Find how many intersections with CT box
            intersections = ray_box_intersection(&V, ts);

            // If there are two intersections, perform the volume ray tracing and save the result to the output image
            if (intersections == 2)
                out_img[d][e] = volume_ray_tracing(V, ts, CT, SHADING);

            // Deallocate the memory used for the ray V
            delete V.a;
            delete V.b;
        }

    // Open the file "outImage.raw" for writing in binary mode
    outfid = fopen("outImage.raw", "wb");

    // Write the output image to the file
    n = fwrite(out_img, sizeof(char), IMG_ROWS * IMG_COLS, outfid);

    // Close the input and output files
    fclose(infid);
    fclose(outfid);

    // Terminate the program with a zero status
    exit(0);

    // Return 0 to indicate successful completion of the function
    return 0;
}