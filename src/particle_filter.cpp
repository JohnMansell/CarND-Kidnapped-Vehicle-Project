#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


// Default Random Engeine
	static default_random_engine gen;


//------------------------------
//      Initialize
//------------------------------
	void ParticleFilter::init(double x, double y, double theta, double std[]) {

		num_particles = 10;

		// define normal distributions for sensor noise
		normal_distribution<double> N_x_init(0, std[0]);
		normal_distribution<double> N_y_init(0, std[1]);
		normal_distribution<double> N_theta_init(0, std[2]);

		// init particles
		for (int i = 0; i < num_particles; i++) {
			Particle p;
			p.id = i;
			p.x = x;
			p.y = y;
			p.theta = theta;
			p.weight = 1.0;

			// add noise
			p.x += N_x_init(gen);
			p.y += N_y_init(gen);
			p.theta += N_theta_init(gen);

			particles.push_back(p);
		}

		is_initialized = true;
	}


//------------------------------
//      Prediction
//------------------------------
	void ParticleFilter::prediction(double delta_t, double sigma_pos[], double velocity, double yaw_rate) {

		// Std Deviation
			double std_x = sigma_pos[0];
			double std_y = sigma_pos[1];
			double std_t = sigma_pos[2];

		// Gaussian Distribution
			normal_distribution<double> dist_x(0, std_x);
			normal_distribution<double> dist_y(0, std_y);
			normal_distribution<double> dist_t(0, std_t);


		for (int i = 0; i < num_particles; ++i) {

			// Initial Position
				double x_old	 = particles[i].x;
				double y_old	 = particles[i].y;
				double theta_old = particles[i].theta;

				double x_fin;
				double y_fin;
				double t_fin;

			// Predict new state
				if (abs(yaw_rate) > 1e-5) {
					// Apply equations of motion model (turning)
					t_fin       = theta_old + ( yaw_rate * delta_t ) ;
					x_fin	    = x_old + velocity / yaw_rate * (sin(t_fin) - sin(theta_old));
					y_fin	    = y_old + velocity / yaw_rate * (cos(theta_old) - cos(t_fin));
				} else {
					// Apply equations of motion model (going straight)
					t_fin = theta_old;
					x_fin	   = x_old + velocity * delta_t * cos(theta_old);
					y_fin	   = y_old + velocity * delta_t * sin(theta_old);
				}

			// Particle = position + noise
				double x_noise = dist_x(gen);
				double y_noise = dist_y(gen);
				double t_noise = dist_t(gen);

				particles[i].x = x_fin + x_noise;
				particles[i].y = y_fin + y_noise;
				particles[i].theta = t_fin + t_noise;
		}
	}


//------------------------------
//      Data Association
//------------------------------
	void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

		// For each observation
			for (auto& obs : observations) {
				double min_dist = numeric_limits<double>::max();

			// Nearest Neighbor
				for (const auto& pred_obs : predicted) {
					double d = dist(obs.x, obs.y, pred_obs.x, pred_obs.y);
					if (d < min_dist) {
						obs.id	 = pred_obs.id;
						min_dist = d;
					}
				}
			}
	}



//------------------------------
//      Update Weights
//------------------------------
	void ParticleFilter::updateWeights( double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

		// For Each Particle
		for (int i = 0; i < num_particles; i++)
		{
			// Particle Position <x, y, t>
				double p_x = particles[i].x;
				double p_y = particles[i].y;
				double p_theta = particles[i].theta;

			// Landmark Positions
				vector<LandmarkObs> predictions;

			// For each map landmark
				for (unsigned int j=0; j < map_landmarks.landmark_list.size(); j++)
				{
					// Landmark <x, y, id>
						float lm_x = map_landmarks.landmark_list[j].x_f;
						float lm_y = map_landmarks.landmark_list[j].y_f;
						int lm_id = map_landmarks.landmark_list[j].id_i;

					// Sensor Range
						double distance = dist(p_x, p_y, lm_x, lm_y);
						if (distance < sensor_range)
						{
							// Add prediction to vector
							predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
						}
				}

			// Transformed Observations -- vehicle(x,y) -> map (x,y)
				vector<LandmarkObs> transformed_os;
				for (unsigned int j = 0; j < observations.size(); j++)
				{
					double x_map = p_x + ( cos(p_theta) * observations[j].x) - ( sin(p_theta) * observations[j].y);
					double y_map = p_y + ( sin(p_theta) * observations[j].x) + ( cos(p_theta) * observations[j].y);

					transformed_os.push_back(LandmarkObs{ observations[j].id, x_map, y_map});
				}

			// Associate predictions to landmarks
				dataAssociation(predictions, transformed_os);

			// Re-init weight
				particles[i].weight = 1.0;

			// Particle Weights
				for (unsigned int j =0; j < transformed_os.size(); j++)
				{
					// Predicted (x,y)
						double pr_x, pr_y;

					// Observed (x,y)
						double o_x = transformed_os[j].x;
						double o_y = transformed_os[j].y;

						int associated_prediction = transformed_os[j].id;

					// Set Predicted (x,y) -- from prediction.id
						for (unsigned int k = 0; k < predictions.size(); k++)
						{
							if (predictions[k].id == associated_prediction)
							{
								pr_x = predictions[k].x;
								pr_y = predictions[k].y;
							}
						}

					// Observation weight -- Multivariate Gaussian
						double sig_x = std_landmark[0];
						double sig_y = std_landmark[1];
						double x_obs = o_x;
						double y_obs = o_y;
						double mu_x  = pr_x;
						double mu_y  = pr_y;

						double gauss_norm = (1/ (2 * M_PI * sig_x * sig_y));
						double exponent =  pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)) + ( pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
						double weight = gauss_norm * exp(-exponent);

					// Total Particle weight
						if (weight != 0)
							particles[i].weight *= weight;

				}

		}

	}


//------------------------------
//      Resample
//------------------------------
	void ParticleFilter::resample() {

		// Particle Weights
			vector<double> particle_weights;
			for (const auto& particle : particles)
				particle_weights.push_back(particle.weight);

		// Weighted Distribution
			discrete_distribution<int> weighted_distribution(particle_weights.begin(), particle_weights.end());

		// Resample Particles
			vector<Particle> resampled_particles;
			for (int i = 0; i < num_particles; ++i) {
				int k = weighted_distribution(gen);
				resampled_particles.push_back(particles[k]);
			}

			particles = resampled_particles;

		// Reset weights for all particles
			for (auto& particle : particles)
				particle.weight = 1.0;
	}


//------------------------------
//      Set Associations
//------------------------------
	Particle ParticleFilter::SetAssociations(Particle& particle,
	                                         const std::vector<int>      & associations,
	                                         const std::vector<double>   & sense_x,
	                                         const std::vector<double>   & sense_y)
	{
		particle.associations.clear();
		particle.sense_x.clear();
		particle.sense_y.clear();

		particle.associations= associations;
		particle.sense_x = sense_x;
		particle.sense_y = sense_y;

		return particle;
	}


//------------------------------
//      Get Associations
//------------------------------
	string ParticleFilter::getAssociations(Particle best)
	{
		vector<int> v = best.associations;
		stringstream ss;
		copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
		string s = ss.str();
		s = s.substr(0, s.length()-1);  // get rid of the trailing space
		return s;
	}

//------------------------------
//      Get Sense X
//------------------------------
	string ParticleFilter::getSenseX(Particle best)
	{
		vector<double> v = best.sense_x;
		stringstream ss;
		copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
		string s = ss.str();
		s = s.substr(0, s.length()-1);  // get rid of the trailing space
		return s;
	}

//------------------------------
//      Get Sense Y
//------------------------------
	string ParticleFilter::getSenseY(Particle best)
	{
		vector<double> v = best.sense_y;
		stringstream ss;
		copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
		string s = ss.str();
		s = s.substr(0, s.length()-1);  // get rid of the trailing space
		return s;
	}
