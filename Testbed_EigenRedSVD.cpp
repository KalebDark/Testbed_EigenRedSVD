// Testbed_EigenRedSVD.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <iostream>
#include <cmath>
#include <Eigen/Core>

#include "RedSVD.h"

// Bearing in mind that normally you'd avoid single letter variables
// If you are using single letter variables, 
// make sure it is well understood that they are standard terms
int main() {
	// Using Eigen
	// Taken from the getting started guide at
	// http://eigen.tuxfamily.org/dox/GettingStarted.html

	// First test on page (see above)
	{
		Eigen::MatrixXd m(2, 2);
		m(0, 0) = 3;
		m(1, 0) = 2.5;
		m(0, 1) = -1;
		m(1, 1) = m(1, 0) + m(0, 1);
		std::cout << m << std::endl;
	}

	// Second test on page (see above)
	{
		Eigen::Matrix3d m = Eigen::Matrix3d::Random();
		m = (m + Eigen::Matrix3d::Constant(1.2)) * 50;
		std::cout << "m =" << std::endl << m << std::endl;
		Eigen::Vector3d v(1, 2, 3);

		std::cout << "m * v =" << std::endl << m * v << std::endl;
	}


	// Using RedSVD-h which is a header only version of RedSVD
	// RedSVD - https://code.google.com/p/redsvd/
	// RedSVD-h - https://github.com/ntessore/redsvd-h

	/* Take from https://code.google.com/p/redsvd/wiki/English
	Inside of redsvd
	The code redsvd.hpp is the core part of redsvd and self explanatory.

	Let A be a matrix to be analyzed with n rows and m columns, and r be the ranks of a truncated SVD 
	(if you choose r = min(n, m), then this is the original SVD).

	First a random Gaussian matrix O with m rows and r columns is sampled and computes Y = At O. 
	Then apply the Gram-Schmidt process to Y so that each column of Y is ortho-normalized. 
	Then we compute B = AY with n rows and r columns. 
	Although the size of Y is much smaller than that of A, Y holds the informatin of A; that is AYYt = A. 
	Intuitively, the row informatin is compresed by Y and can be decompressed by Yt

	Similarly, we take another random Gaussian matrix P with r rows and r columns, and compute Z = BP. 
	As in the previous case, the columns of Z are ortho-normalized by the Gram-Schmidt process. ZZtt B = B. 
	Then compute C = Zt B.

	Finally we compute SVD of C using the traditional SVD solver, 
	and obtain C = USVt where U and V are orthonormal matrices, 
	and S is the diagonal matrix whose entriesa are singular values. 
	Since a matrix C is very small, this time is negligible.

	Now A is decomposed as A = AYYt = BYt = ZZtBYt = ZCYt = ZUSVtYt. 
	Both ZU and YV are othornormal, and ZU is the left singular vectors and YV is the right singular vector.
	S is the diagonal matrix with singular values.
	*/

	const int numberOfRows = 100;
	const int numberOfColumns = 100;
	const int actualRank = 20;
	const int estimateRank = 10;

	std::cout << "Dense matrix test " << numberOfRows << "\t" << numberOfColumns << "\t" << actualRank << "\t" << estimateRank << std::endl;

	Eigen::MatrixXf U = Eigen::MatrixXf::Random(numberOfRows, actualRank);
	Eigen::MatrixXf V = Eigen::MatrixXf::Random(numberOfRows, actualRank);

	RedSVD::GramSchmidt<Eigen::MatrixXf>(U);
	RedSVD::GramSchmidt<Eigen::MatrixXf>(V);

	Eigen::VectorXf S(actualRank);
	for (int i = 0; i < actualRank; ++i) {
		S(i) = std::pow(0.9f, i);
	}

	Eigen::MatrixXf A = U * S.asDiagonal() * V.transpose();
	RedSVD::RedSVD<Eigen::MatrixXf> svdOfA(A, estimateRank);

	for (int i = 0; i < estimateRank; ++i) {
		std::cout << i << "\t" << std::log(S(i)) << "\t" << std::log(svdOfA.singularValues()(i)) << "\t"
			<< std::fabs(U.col(i).dot(svdOfA.matrixU().col(i))) << "\t"
			<< std::fabs(V.col(i).dot(svdOfA.matrixV().col(i))) << std::endl;
	}

	return 0;
}

