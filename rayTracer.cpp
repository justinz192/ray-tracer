#include <limits>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include "geometry.h"

using namespace std;

struct Light {
    Light(const Vec3f &p, const float &i) : position(p), intensity(i){}
    Vec3f position;
    float intensity;
};

struct Material {
    //constructor initializing material color
    Material(const Vec3f &a, const Vec3f &color, const float &spec) : albedo(a), diffuse_color(color), specular_exponent(spec) {} 
    Material() : albedo(1, 0, 0), diffuse_color(), specular_exponent() {}
    Vec3f albedo; //the fraction of light that a surface reflects. 
    //If it is all reflected, the albedo is equal to 1. If 30% is reflected, the albedo is 0.3
    //albedo[0] = light reflected as diffuse light, albedo[1] = light reflected as specular light
    Vec3f diffuse_color;
    float specular_exponent; //how shiny or glossy the material is, higher value = sharper and brighter highlight
};

struct Sphere{
    Vec3f center;
    float radius;
    Material material;

    Sphere(const Vec3f &c, const float &r, const Material &m) : center(c), radius(r), material(m) {}
    
    //determines if a ray intersects a sphere
    bool ray_intersect(const Vec3f& orig, const Vec3f &dir, float& t0) const { 
        Vec3f L = center - orig; //vector from the ray's origin (orig) to the sphere's center (center).
        float tca = L*dir; //projected distance of the sphere's center onto the ray's direction vector
        float d2 = L*L - tca*tca; //squared perpendicular distance from the sphere's center to the ray
        if (d2 > radius*radius) return false; //if that distance is longer than the squared radius, ray misses the sphere
        float thc = sqrtf(radius*radius-d2); // half the length of the chord (intersection line) through the sphere
        
        //find distances along ray direction to two intersection points
        t0 = tca - thc; //distance to closer intersection point
        float t1 = tca + thc; //distance to further intersection point
        if (t0 < 0) t0 = t1; //if t0 < 0, then closer intersection is behind the ray's origin
        //this means the ray starts inside the sphere
        if (t0 < 0) return false; //if t0 (now t1) < 0, ray doesn't intersect sphere in front of camera
        return true; //if it checks all passes, the ray does intersect the sphere
        //t0 contains the distance to the closest intersection point
    }
};

//calculates the reflection vector of a given light ray off a surface with normal N
Vec3f reflect(const Vec3f &I, const Vec3f &N) { 
    return I - N*2.f*(I*N); //uses reflection formula: R=I−2⋅(I⋅N)⋅N
}

//loops through all spheres and finds the closest intersection
bool scene_intersect(const Vec3f &orig, const Vec3f &dir, const vector<Sphere> &spheres, Vec3f &hit, Vec3f &N, Material &material) {
    float spheres_dist = numeric_limits<float>::max(); //keeps track of distance of closest intersection
    for (size_t i=0; i < spheres.size(); i++) {
        float dist_i;
        if (spheres[i].ray_intersect(orig, dir, dist_i) && dist_i < spheres_dist) { 
            //if a ray intersects a sphere and is the closest hit sphere
            spheres_dist = dist_i;//update closest intersection
            hit = orig + dir*dist_i; //determines the coordinates of the intersection
            N = (hit - spheres[i].center).normalize(); //computes the normal unit vector at the intersection
            material = spheres[i].material; //assigns the material of the sphere that was hit
        }
    }
    return spheres_dist<1000; //if intersection was found, this returns true
}

//function to determine the color of a ray after it interacts with a sphere
Vec3f cast_ray(const Vec3f &orig, const Vec3f &dir, const vector<Sphere> &spheres, const vector<Light> &lights, size_t depth=0) {
    Vec3f point, N;
    Material material;

    if (depth>4 || !scene_intersect(orig, dir, spheres, point, N, material)) { //if ray does not intersect
        return Vec3f(0.2, 0.7, 0.8); // background color
    }

    Vec3f reflect_dir = reflect(dir, N).normalize(); 
    Vec3f reflect_orig = reflect_dir*N < 0 ? point - N*1e-3 : point + N*1e-3; // offset the original point to avoid occlusion by the object itself
    Vec3f reflect_color = cast_ray(reflect_orig, reflect_dir, spheres, lights, depth + 1);

    //Determines diffuse illumination
    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (size_t i =0; i < lights.size(); i++) {
        Vec3f light_dir = (lights[i].position - point).normalize(); //computes direction from the intersection to light source
        float light_distance = (lights[i].position - point).norm();

        Vec3f shadow_orig = light_dir*N < 0 ? point - N*1e-3 : point + N*1e-3; // checking if the point lies in the shadow of the lights[i]
        Vec3f shadow_pt, shadow_N;
        Material tmpmaterial;
        if (scene_intersect(shadow_orig, light_dir, spheres, shadow_pt, shadow_N, tmpmaterial) && (shadow_pt-shadow_orig).norm() < light_distance)
            continue;

        //determine the intensity of the light using dot product between light direction and surface normal
        diffuse_light_intensity += lights[i].intensity * max(0.f, light_dir*N); 
        
        //determines the specular intensity (shiny highlights) using dot product between light direction and reflected direction
        //raised to the specular exponent to control how sharp or blurry highlight is
        //scaled by light intensity
        specular_light_intensity += powf(max(0.f, -reflect(-light_dir, N)*dir), 
        material.specular_exponent)*lights[i].intensity;
    }
    //if intersects with a sphere, returns sum of diffuse and specular lighting values
    //scaled by light intensity and albedo value (specular highlights are white (1., 1., 1.,) or (255, 255, 255))
    return material.diffuse_color*diffuse_light_intensity*material.albedo[0] + Vec3f(1., 1., 1.)*specular_light_intensity * material.albedo[1] + reflect_color*material.albedo[2]; 
}

//renders and outputs the color of each pixel to ppm file
void render(const vector<Sphere> &spheres, const vector<Light> &lights) {
    const int width    = 1024;
    const int height   = 768;
    const int fov      = M_PI/3.;
    vector<Vec3f> framebuffer(width*height);//1D array storing each pixel colour

    #pragma omp parallel for //enables multi-threading to improve performance
    for (size_t j = 0; j<height; j++) { //loops through each row
        for (size_t i = 0; i<width; i++) { //loops through each pixel in row
            //converts pixel (i, j) into normalized coordinate system
            float x =  (2*(i + 0.5)/(float)width  - 1)*tan(fov/2.)*width/(float)height;
            float y = -(2*(j + 0.5)/(float)height - 1)*tan(fov/2.);
            Vec3f dir = Vec3f(x, y, -1).normalize(); //determines direction of ray starting at (0, 0, 0) 
            //dir points towards each pixel in the -z direction
            framebuffer[i+j*width] = cast_ray(Vec3f(0,0,0), dir, spheres, lights); //determines if ray intersects a sphere
            //returns the color of the sphere (considering light rays) if hit, otherwise returns color of background
        }
    }

    ofstream ofs; // save the framebuffer to file
    ofs.open("./out.ppm",ios::binary); //opens ppm file in binary
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < height*width; ++i) {
        //ensure framebuffer color values are within the right range (0.0 to 1.0)
        Vec3f &c = framebuffer[i];
        float myMax = max(c[0], max(c[1], c[2])); //finds biggest value of RGB for pixel
        if (myMax > 1) c = c*(1./myMax); //if larger than 1.0, normalize to proper range
        for (size_t j = 0; j<3; j++) {
            ofs << (char)(255 * framebuffer[i][j]); //converts color ranges (0.0 to 1.0) to 8-bit RGB values (0 to 255)
        }
    }
    ofs.close();
}

int main() {
    Material      ivory(Vec3f(0.6, 0.3, 0.1), Vec3f(0.4, 0.4, 0.3), 50.);
    Material red_rubber(Vec3f(0.9, 0.1, 0.0), Vec3f(0.3, 0.1, 0.1), 10.);
    Material     mirror(Vec3f(0.0, 10.0, 0.8), Vec3f(1.0, 1.0, 1.0), 1425.);
    
    vector<Sphere> spheres;
    spheres.push_back(Sphere(Vec3f(-3,    0,   -16), 2,      ivory));
    spheres.push_back(Sphere(Vec3f(-1.0, -1.5, -12), 2,      mirror));
    spheres.push_back(Sphere(Vec3f( 1.5, -0.5, -18), 3,      red_rubber));
    spheres.push_back(Sphere(Vec3f( 7,    5,   -18), 4,      mirror));

    vector<Light>  lights;
    lights.push_back(Light(Vec3f(-20, 20,  20), 1.5));
    lights.push_back(Light(Vec3f( 30, 50, -25), 1.8));
    lights.push_back(Light(Vec3f( 30, 20,  30), 1.7));

    render(spheres, lights);

    return 0;
}