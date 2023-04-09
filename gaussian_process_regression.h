#ifndef GPR_H
#define GPR_H


#include <Eigen/Dense>
#include <iostream>
#include <vector>

//#ifdef USE_DOUBLE_PRECISION
//typedef double REALTYPE;
//#else
//typedef float REALTYPE;
//#endif




template<typename REALTYPE>
class GaussianProcessRegression {

	typedef Eigen::Matrix<REALTYPE, Eigen::Dynamic, Eigen::Dynamic> MatrixXr;
	typedef Eigen::Matrix<REALTYPE, Eigen::Dynamic, 1> VectorXr;

	/*Of course, Eigen is not limited to matrices whose dimensions are known at compile time. The RowsAtCompileTime and ColsAtCompileTime template parameters can take the special value Dynamic which indicates that the size is unknown at compile time, so must be handled as a run-time variable.*/

	/*
	
	The three mandatory template parameters of Matrix are:

Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
Scalar is the scalar type, i.e. the type of the coefficients. That is, if you want a matrix of floats, choose float here. See Scalar types for a list of all supported scalar types and for how to extend support to new types.
RowsAtCompileTime and ColsAtCompileTime are the number of rows and columns of the matrix as known at compile time (see below for what to do if the number is not known at compile time).
	*/

	MatrixXr input_data_;
	MatrixXr output_data_;
	MatrixXr KXX;
	MatrixXr KXX_;
	VectorXr KXx;
	//MatrixXr KxX;

	int n_data_;
	bool b_need_prepare_;

	double l_scale_;
	double sigma_f_;
	double sigma_n_;

	VectorXr dist;

	VectorXr regressors;

	//  std::vector<Eigen::FullPivLU<MatrixXr> > decompositions_;
	MatrixXr alpha_;

public:
	GaussianProcessRegression(int inputDim, int outputDim);
	/*
	l_scale_: la longitud de escala de la covarianza. Controla la distancia característica a la que las entradas del modelo tienen una influencia significativa entre sí. Un valor más grande de l_scale_ indica una mayor correlación entre entradas lejanas y una superficie de respuesta más suave.

	sigma_f_: la desviación estándar de la función de covarianza. Controla la variación general de la superficie de respuesta. Un valor más grande de sigma_f_ indica una superficie de respuesta más variable.

	sigma_n_: la desviación estándar del ruido de observación. Controla la cantidad de error aleatorio presente en las observaciones del modelo. Un valor más grande de sigma_n_ indica que las observaciones son más ruidosas.
	
	*/

	void SetHyperParams(double l, double f, double n) { l_scale_ = l; sigma_f_ = f; sigma_n_ = n; };
	void GetHyperParams(double& l, double& f, double& n) { l = l_scale_; f = sigma_f_; n = sigma_n_; };

	// add data one by one
	void AddTrainingData(const VectorXr& newInput, const VectorXr& newOutput);
	// batch add data
	void AddTrainingDataBatch(const MatrixXr& newInput, const MatrixXr& newOutput);

	REALTYPE SQEcovFuncD(VectorXr x1, VectorXr x2);
	void Debug();

	MatrixXr SQEcovFunc(MatrixXr x1);
	VectorXr SQEcovFunc(MatrixXr x1, VectorXr x2);
		MatrixXr KernelCovFunc(const MatrixXr & x1, const MatrixXr & x2, const Eigen::MatrixXd & kernel);


	// these are fast methods 
	void PrepareRegression(bool force_prepare = false);
	VectorXr DoRegression(const VectorXr& inp, Eigen::MatrixXd kernel, bool prepare = false);
	// these are the old implementations that are slow, inaccurate and easy to understand
	void PrepareRegressionOld(bool force_prepare = false);
	VectorXr DoRegressionOld(const VectorXr& inp, bool prepare = false);

	int get_n_data() { return n_data_; };
	const MatrixXr& get_input_data() { return input_data_; };
	const MatrixXr& get_output_data() { return output_data_; };

	void ClearTrainingData();


};


#include "gaussian_process_regression.hxx"
#endif //GPR_H