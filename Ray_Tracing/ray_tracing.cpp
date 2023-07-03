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


/*
This function calculates the intersection of a ray with a sphere. It takes a Ray and SPHERE as input, along with pointers 
for a Point3d to store the normal of the sphere at the intersection, a Point3d to store the intersection point, 
and a float pointer to store the material property of the sphere. It returns the distance from the origin of the ray 
to the intersection point, or 0 if there is no intersection. The function first calculates the quadratic coefficients A, B, 
and C of the intersection equation, and then checks whether the discriminant D is positive. 
If D is positive, the function calculates the intersection point, the normal of the sphere at the intersection point, 
and the material property, and returns the distance from the origin of the ray to the intersection point. 
If D is not positive, the function returns 0.
*/
float ray_sphere_intersection(Ray ray, SPHERE sphere, Point3d* n1, Point3d* intersection_point1, float* kd1) {
  float t;
  float radius;
  Point3d center;
  Point3d S;

  *kd1 = sphere.kd;
  radius = sphere.radius;
  center.x = sphere.x;
  center.y = sphere.y;
  center.z = sphere.z;
  sub_vector(&center, ray.a, &S);

  float A = dot_product(ray.b, ray.b);
  float B = -2.0 * dot_product(&S, ray.b);
  float C = dot_product(&S, &S) - radius * radius;
  float D = B * B - 4 * A * C;

  if (D >= 0.0) {
    int sign = (C < 0.0) ? 1 : -1;
    t = (-B + sign * sqrt(D)) / 2.0;

    n1->x = (ray.a->x + ray.b->x * t - center.x) / radius;
    n1->y = (ray.a->y + ray.b->y * t - center.y) / radius;
    n1->z = (ray.a->z + ray.b->z * t - center.z) / radius;

    intersection_point1->x = ray.a->x + ray.b->x * t;
    intersection_point1->y = ray.a->y + ray.b->y * t;
    intersection_point1->z = ray.a->z + ray.b->z * t;

    return t;
  }
  return 0.0;
}



// Find the index of the maximum absolute value of an array of floats
int find_max(const float a[3]) {
    // Initialize the maximum absolute value and index to the first element
    float max = fabsf(a[0]);
    int index = 0;

    // Loop over the remaining elements to find the maximum absolute value and its index
    for (int i = 1; i < 3; i++) {
        const float temp = fabsf(a[i]);
        if (temp > max) {
            max = temp;
            index = i;
        }
    }

    // Return the index of the maximum absolute value
    return index;
}


/*
This function takes in an array of length 3 representing the surface normal and a 2D array of size 4x3 representing the 4 vertices 
of a quadrilateral in 3D space, and a 2D array of size 4x2 that will contain the 2D coordinates of the vertices
projected onto a plane perpendicular to the surface normal
*/
void find_project_plane(float surface_normal[3], float vertices[4][3], float projected_vertices[4][2])
{

// Determine which index of the surface normal array has the maximum absolute value
int max_normal_index = find_max(surface_normal);

// Loop through the four vertices and assign their projected x and y coordinates
// based on the index of the maximum absolute value in the surface normal
for (int i = 0; i < 4; i++) {
    switch (max_normal_index) {
        // If the x-coordinate has the maximum absolute value, project onto the y-z plane
        case 0:
            projected_vertices[i][0] = vertices[i][1];
            projected_vertices[i][1] = vertices[i][2];
            break;
        
        // If the y-coordinate has the maximum absolute value, project onto the x-z plane
        case 1:
            projected_vertices[i][0] = vertices[i][0];
            projected_vertices[i][1] = vertices[i][2];
            break;
        
        // If the z-coordinate has the maximum absolute value, project onto the x-y plane
        case 2:
            projected_vertices[i][0] = vertices[i][0];
            projected_vertices[i][1] = vertices[i][1];
            break;
    }
}
}

//Find projection point
void findProjPoint(float a[3], Point3d* p, float dp[2])
{
int index;
index = find_max(a);

if (index == 0)
{
	dp[0] = p->y;
	dp[1] = p->z;
}
else if (index == 1)
{
	dp[0] = p->x;
	dp[1] = p->z;
}
else if (index == 2)
{
	dp[0] = p->x;
	dp[1] = p->y;
}
}


/*
This function implements the point-in-polygon test, which checks whether a given point is inside or outside of a polygon defined by a set of vertices.
It takes as input the number of vertices of the polygon (number_of_vert), arrays of the x and y coordinates of the vertices (vert_x and vert_y), 
and the x and y coordinates of the test point (test_x and test_y).
The function then loops through each edge of the polygon and checks whether the test point is to the left or to the right of the edge. 
If the test point is to the left of an odd number of edges, then it is inside the polygon and the function returns 1. 
If the test point is to the left of an even number of edges, then it is outside the polygon and the function returns 0.
*/
int point_in_polygon_test(int number_of_vertices, float *vertex_x, float *vertex_y, float test_x, float test_y) {
    int inside = 0; // Initialize point-inside-polygon status to "false"
    for (int i = 0, j = number_of_vertices - 1; i < number_of_vertices; j = i++) {
        // Check if the ith vertex is above the point and the jth vertex is below the point,
        // or vice versa, along the y-axis, and if the point's x-coordinate falls within the
        // x-coordinate range of the edge defined by vertices i and j.
        if ((vertex_y[i] > test_y) != (vertex_y[j] > test_y) &&
            test_x < (vertex_x[j] - vertex_x[i]) * (test_y - vertex_y[i]) / (vertex_y[j] - vertex_y[i]) + vertex_x[i]) {
            inside = !inside; // Flip the point-inside-polygon status
        }
    }
    return inside; // Return the final point-inside-polygon status
}



/*
This function calculates the intersection of a ray and a polygon with four vertices. The function takes as input a Ray 
and a POLY4 structure (containing the coordinates of the four vertices and the normal vector of the polygon) 
and returns the distance between the ray origin and the intersection point, or 0 if there is no intersection. The function also returns 
the surface normal of the polygon, the intersection point, and the surface's diffusion coefficient. It first calculates 
the value of D and the intersection point of the ray and the polygon. It then projects the intersection point onto a 
plane defined by the polygon's normal vector, and checks if the projected point lies within the polygon. If it does, the function returns 
the distance between the ray origin and the intersection point. Otherwise, it returns 0.
*/
float ray_polygon_intersection(Ray ray, POLY4 poly, Point3dPtr n2, Point3dPtr intersection_point2, float *kd2) {
    float t, temp1, temp2, D;
    float projectPlane[4][2], projPoint[2];
    float p0[2], p1[2], p2[2], p3[2], Array[4];
    int c;
    Point3d np;

    // Calculate the distance from the origin to the polygon
    D = -1 * (poly.N[0] * poly.v[0][0] + poly.N[1] * poly.v[0][1] + poly.N[2] * poly.v[0][2]);
    
    // Calculate the intersection point
    np = assign_values(poly.N);
    temp1 = dot_product(&np, ray.a) + D;
    temp2 = dot_product(&np, ray.b);
    t = -1 * (temp1 / temp2);

    // Store the diffuse coefficient of the polygon
    *kd2 = poly.kd;

    // Store the surface normal of the polygon
    n2->x = poly.N[0];
    n2->y = poly.N[1];
    n2->z = poly.N[2];

    // Store the intersection point
    intersection_point2->x = ray.a->x + ray.b->x*t;
    intersection_point2->y = ray.a->y + ray.b->y*t;
    intersection_point2->z = ray.a->z + ray.b->z*t;

    // Get the projection of the intersection point onto the polygon
    findProjPoint(poly.N, intersection_point2, projPoint);

    // Get the projection plane of the polygon
    find_project_plane(poly.N, poly.v, projectPlane);

    // Test if the projected point is inside the polygon
    for(int i = 0; i < 4; i++){
        p0[i] = projectPlane[i][0];
        p1[i] = projectPlane[i][1];
    }
    c = point_in_polygon_test(4, p0, p1, projPoint[0], projPoint[1]);
    if ((temp2 > 0.0f) || (temp2 == 0.0f))
        return 0.0f;  	// Ray and Polygon parallel, intersection rejection
    else {
        if (c != 0)
            return t;	// The distance to the intersection point
        else
            return 0.0f;
    }
}



/*
This function calculates the intersection point between a given ray and either a sphere or a polygon object in a 3D scene. 
It returns the distance between the origin of the ray and the intersection point, along with the normal of the object at the intersection point 
and the intersection_pointolated color of the object at that point.
It first checks for an intersection with a sphere and a polygon object, respectively, using two separate helper functions. 
If neither intersection occurs, the function returns 0.0, indicating no intersection. If only one intersection occurs, the function sets the normal
and intersection point to the values returned by the corresponding helper function and returns the distance to the intersection. 
If both intersections occur, the function returns the intersection with the closest distance, based on which intersection occurred first. 
The function also assigns the normal and intersection point of the closest intersection to the output parameters n and intersection_point and the intersection_pointolated color to kd.

Compute the intersection of a ray with an object, which can be a sphere or a polygon
Return the distance from the origin of the ray to the intersection point, or 0 if there is no intersection
n is the surface normal at the intersection point
kd is the diffuse coefficient of the object
*/
float ray_object_intersection(const Ray& ray, const SPHERE& sphere, const POLY4& poly, Point3dPtr n, Point3dPtr intersection_point, float* kd)
{
    // Create unique pointers to store the values of the intersection tests for the sphere and polygon
    std::unique_ptr<float> kd1(new float);
    std::unique_ptr<float> kd2(new float);
    std::unique_ptr<Point3d> n1(new Point3d);
    std::unique_ptr<Point3d> n2(new Point3d);
    std::unique_ptr<Point3d> intersection_point1(new Point3d);
    std::unique_ptr<Point3d> intersection_point2(new Point3d);

    // Compute the intersection of the ray with the sphere and polygon
    float t1 = ray_sphere_intersection(ray, sphere, n1.get(), intersection_point1.get(), kd1.get());
    float t2 = ray_polygon_intersection(ray, poly, n2.get(), intersection_point2.get(), kd2.get());

    // If the ray does not intersect with either object, return 0
    if (t1 == 0.0f && t2 == 0.0f)
    {
        return 0.0f;
    }

    // If the ray only intersects with the sphere, return the intersection distance and set the surface normal, intersection point, and diffuse coefficient to the sphere values
    else if (t2 == 0.0f)
    {
        *n = *n1;
        *intersection_point = *intersection_point1;
        *kd = *kd1;
        return t1;
    }

    // If the ray only intersects with the polygon or intersects with both objects but the intersection with the sphere is farther away, return the intersection distance and set the surface normal, intersection point, and diffuse coefficient to the polygon values
    else if (t1 == 0.0f || t1 < t2)
    {
        *n = *n1;
        *intersection_point = *intersection_point1;
        *kd = *kd1;
        return t1;
    }

    // If the ray intersects with both objects and the intersection with the polygon is closer, return the intersection distance and set the surface normal, intersection point, and diffuse coefficient to the polygon values
    else
    {
        *n = *n2;
        *intersection_point = *intersection_point2;
        *kd = *kd2;
        return t2;
    }
}



/*
This function takes in the light source position lrp, the normal vector n, the intersection point p, 
the diffuse coefficient kd, and the intensity of the light ip. It calculates the dot product between the normalized light vector 
and the normal vector to obtain the cosine of the angle between them, which is then multiplied by the diffuse coefficient 
and the light intensity to obtain the shading value. If the cosine is negative, the shading value is set to zero, 
indicating that the surface is facing away from the light and therefore not illuminated. The shading value is then returned as an unsigned char.
*/
int shading(float lrp[3], Point3d* n, Point3d* p, float* kd, float ip) {
    Point3d l, ln;
    float temp = 0.0f;
    unsigned char C = 0;

    // Calculate the vector from the point to the light and normalize it
    l.x = lrp[0] - p->x;
    l.y = lrp[1] - p->y;
    l.z = lrp[2] - p->z;
    normalize(&l, &ln);

    // Calculate the dot product of the surface normal and the normalized light vector
    temp = dot_product(n, &ln);

    // If the angle between the surface normal and the light vector is less than 90 degrees, the face is illuminated
    // Otherwise, it is in shadow and the color value is 0
    if (temp < 0)
        temp = 0;

    // Calculate the color value based on the diffuse reflectivity coefficient and the light intensity
    C = (unsigned char)(int)ip * (*kd) * temp;
    return C;
}

/*
The function ray_tracing takes a Ray and two shapes, a SPHERE and a POLY4, as well as a light position lrp and intensity ip. 
The function first computes the intersection point P, normal vector n, and intersection_pointolation point intersection_point of the ray with the first object 
it intersects (either the sphere or the polygon). If the intersection is valid (i.e., P is nonzero), the function computes the shading 
color C at the intersection point using the shading function and returns C. If the intersection is not valid, the function returns 0.
*/
int ray_tracing(Ray ray, SPHERE sphere, POLY4 poly, float lrp[3], float ip) {
    float P;
    Point3d n;
    Point3d intersection_point;

    int C;
    float kd;

    P = ray_object_intersection(ray, sphere, poly, &n, &intersection_point, &kd);

    if (P != 0.0) {
        C = shading(lrp, &n, &intersection_point, &kd, ip);
        return C;
    }
    else
        return 0;
}



/*
This is the main function that sets up the scene from mymodel.h and performs ray tracing for every pixel in the image plane. 
It initializes transformation matrices and constructs rays for each pixel. Then, it calls the ray tracing function 
to determine the color at the intersection point of each ray with the scene. The results are stored in an image buffer, 
which is then written to a binary file.
*/
int main(int argc, char* argv[]) {

  // Declare variables and initialize them with predefined values
	int i, j;
	int c;
	transform3d TIN1, RT1;
	transform3d Mcw;
	Point3d Pvpn, Pvup, Pvrp;

	Pvpn = assign_values(VPN);
	Pvup = assign_values(VUP);
	Pvrp = assign_values(VRP);

  // Calculate the inverse translation and transpose rotation matrices
	translation_inverse(VRP, TIN1);
	rotation_transpose(&Pvpn, &Pvup, RT1);

  // Calculate the final transformation matrix
	mult_transforms(TIN1, RT1, Mcw);

	// Initialize the image buffer with black
	for (i = 0; i < ROWS; i++)
	{
		for (j = 0; j < COLS; j++)
		{
			img[i][j] = 0;
		}
	}

  // Traverse the image grid and cast rays
	for (i = 0; i < ROWS; i++)
	{
		for (j = 0; j < COLS; j++)
		{
      // Construct ray V for each pixel
			Ray V;
			V.a = new Point3d;
			V.b = new Point3d;
			ray_construction(i, j, focal, xmin, xmax, ymin, ymax, Mcw, &Pvrp, &V);

      // Trace the ray and get the color for that pixel
			c = ray_tracing(V, obj1, obj2, LPR, Ip);
      if (c != 0){
			img[i][j] = c;
    }
      // Delete the allocated memory for the ray
			delete V.a;
			delete V.b;
		}
	}

	// Save the final image as a binary file
    FILE * fp;
	fp = fopen("Ray_default2.raw", "wb");
	fwrite(img, sizeof(unsigned char), sizeof(img), fp);
	fclose(fp);
    return 0;
}
