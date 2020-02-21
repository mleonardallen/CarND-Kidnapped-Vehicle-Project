
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>

#include "particle_filter.h"


using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    num_particles_ = 10;

    // gaussian distributions
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Note: weights are set within updateWeights
    weights_.resize(num_particles_);

    // Initialize particles
    for (int i = 0; i < num_particles_; ++i) {
        
        Particle particle;

        particle.x = dist_x(generator_);
        particle.y = dist_y(generator_);
        particle.theta = dist_theta(generator_);
        particle.weight = 1.0;

        particles.push_back(particle);
    }

    is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    // gaussian noise setup
    normal_distribution<double> noise_x(0, std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);

    // avoid division by zero
    if (fabs(yaw_rate) < 0.0001) {
        yaw_rate = 0.0001;
    }

    // apply the bicycle motion model with gaussian noise
    for (auto&& particle : particles) {
        particle.x += noise_x(generator_) + (velocity / yaw_rate) * (sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta));
        particle.y += noise_y(generator_) + (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t));
        particle.theta += noise_theta(generator_) + yaw_rate * delta_t;
    }
}

// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
// more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
void ParticleFilter::updateWeights(
    double sensor_range, 
    double std_landmark[],
    std::vector<LandmarkObs> observations, 
    Map map_landmarks
) {
    for (int i = 0; i < num_particles_; i++) {

        long double weight = 1.0;

        for (int j = 0; j < observations.size(); j++) {
            LandmarkObs transformed = getTransformedObservation(observations[j], particles[i]);
            Map::single_landmark_s nearest = getNearestLandmark(transformed, map_landmarks, sensor_range);

            // Note: Only kernel is needed for resampling to work
            // Note: Assumes zero correlation between X and Y
            weight *= exp(-0.5 * (
                pow(nearest.x_f - transformed.x, 2) / (std_landmark[0] * std_landmark[0]) + 
                pow(nearest.y_f - transformed.y, 2) / (std_landmark[1] * std_landmark[1])
            ));
        }

        particles[i].weight = weight;
        weights_[i] = weight;
    }
}

// Resample particles with replacement with probability proportional to their weight. 
// NOTE: You may find std::discrete_distribution helpful here.
//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
void ParticleFilter::resample() {

    vector<Particle> resampled_particles;

    discrete_distribution<> distribution(weights_.begin(), weights_.end());
    
    for (int i = 0; i < num_particles_; i++) {
        int idx = distribution(generator_);
        resampled_particles.push_back(particles[idx]);
    }

    particles = resampled_particles;
}

// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
//   according to the MAP'S coordinate system. You will need to transform between the two systems.
//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
//   The following is a good resource for the theory:
// https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
//   and the following is a good resource for the actual equation to implement (look at equation 
//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
//   for the fact that the map's y-axis actually points downwards.)
//   http://planning.cs.uiuc.edu/node99.html
LandmarkObs ParticleFilter::getTransformedObservation(LandmarkObs observation, Particle particle) {
    LandmarkObs meas;

    double cos_theta = cos(particle.theta);
    double sin_theta = sin(particle.theta);

    meas.x = observation.x * cos_theta - observation.y * sin_theta + particle.x;
    meas.y = observation.y * cos_theta + observation.x * sin_theta + particle.y;

    return meas;
}

Map::single_landmark_s ParticleFilter::getNearestLandmark(
    LandmarkObs observation,
    Map map_landmarks,
    double sensor_range
) {
    Map::single_landmark_s nearest;

    double distance;
    double min_distance = sensor_range;
    for (auto&& landmark : map_landmarks.landmark_list) {
        distance = sqrt(pow(landmark.x_f - observation.x, 2) + pow(landmark.y_f - observation.y, 2));
        if (distance < min_distance) {
            min_distance = distance;
            nearest = landmark;
        }
    }

    return nearest;
}

void ParticleFilter::write(std::string filename) {
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles_; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}
