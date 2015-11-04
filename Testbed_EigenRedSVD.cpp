// Testbed_EigenRedSVD.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <chrono>
#include <omp.h>

#define EIGEN_RUNTIME_NO_MALLOC
#define EIGEN_NO_AUTOMATIC_RESIZING

#include <Eigen/Core>
#include <Eigen/Dense>

#include "RedSVD.h"

class Stopwatch {
public:
	void Start() {
		_start = _clock::now();
	};

	double Stop() const {
		return std::chrono::duration_cast<_seconds>(_clock::now() - _start).count();
	};

public:
	Stopwatch() {};

private:
	Stopwatch& operator=(const Stopwatch& other_) = delete;
	Stopwatch(const Stopwatch& other_) = delete;

private:
	typedef std::chrono::steady_clock _clock;
	typedef std::chrono::duration<double, std::ratio<1>> _seconds;
	std::chrono::time_point<_clock> _start;
};

class Sumulation {
public:
	static void TimeIterations(int numberOfIterations_, int numberOfRows_, int numberOfColumns_) {
		Stopwatch timer;
		double totalTime = 0.0;
		for (int iteration = 0; iteration < numberOfIterations_; ++iteration) {
			timer.Start();
			RunIteration(numberOfRows_, numberOfColumns_);
			double timeTaken = timer.Stop();
			totalTime = timeTaken;
			std::cout << "Time take for run: " << timeTaken << "seconds" << std::endl;
		}
		double averageTime = totalTime / numberOfIterations_;
		std::cout << "Average time take for " << numberOfIterations_ << " runs:" << std::endl;
		std::cout << averageTime << "seconds" << std::endl;
		std::cout << std::endl << std::endl;
	}

private:
	static void RunIteration(int numberOfRows_, int numberOfColumns_) {
		std::cout << "------------- Eigen Library Test -------------" << std::endl;
		std::cout << "Number Of Rows " << numberOfRows_ << "\tNumber of Columns " << numberOfColumns_ << std::endl;

		std::cout << "Making U & V" << std::endl;
		Eigen::MatrixXf U = Eigen::MatrixXf::Random(numberOfRows_, numberOfColumns_);
		Eigen::MatrixXf V = Eigen::MatrixXf::Random(numberOfRows_, numberOfColumns_);

		std::cout << "Diagonalising S" << std::endl;
		Eigen::VectorXf S = Eigen::VectorXf::Random(numberOfColumns_);
		Eigen::MatrixXf sDiagonal = S.asDiagonal();

		std::cout << "Transposing V" << std::endl;
		Eigen::MatrixXf vTransposed = V.transpose();

		std::cout << "Building A" << std::endl;
		Eigen::MatrixXf A;

		std::cout << "...using A = S(daigonal) * U * V(transposed)" << std::endl;
		A.noalias() = U * sDiagonal * vTransposed;

		/*std::cout << "Calculating SVD of U.S.V'" << std::endl;
		Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);*/
	};

private:
	Sumulation() = delete;
	Sumulation& operator=(const Sumulation& other_) = delete;
	Sumulation(const Sumulation& other_) = delete;
};

int main() {
	const int numberOfRows = 11344;
	const int numberOfColumns = 513;
	Stopwatch timer;

	int ompMaxThreads = omp_get_max_threads();
	std::cout << "OMP Max number of threads: " << ompMaxThreads << std::endl;
	omp_set_num_threads(ompMaxThreads);
	Eigen::setNbThreads(ompMaxThreads);
	std::cout << "Eigen number of threads: " << Eigen::nbThreads() << std::endl;

	Eigen::internal::set_is_malloc_allowed(false);

	Sumulation::TimeIterations(100, 100, 100);
	Sumulation::TimeIterations(100, 1000, 100);
	Sumulation::TimeIterations(100, 100, 1000);

	std::cout << "Press any key to quit!" << std::endl;
	std::cin.get();

	return 0;
}

