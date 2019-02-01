#include "fittingAlgorithm.h"

vector<double> LMSFit::FitLine(vector<cv::Point> _pt_list,uint _poly_order)
{
	mat X = arma::zeros<mat>(_pt_list.size(),1);
	mat Y = arma::zeros<mat>(_pt_list.size(),_poly_order + 1);
	mat coeff = arma::zeros<mat>(_poly_order + 1,1);

	for(uint i = 0; i < _pt_list.size();i++)
	{
		X(i,0) = _pt_list[i].x;

		Y(i,0) = 1;

		for(uint j = 1; j < _poly_order + 1;j++)
		{
			Y(i,j) = pow(_pt_list[i].y,j);
		}
	}

	coeff = arma::solve(Y,X);

	vector<double> coeff_return;

	for(uint i = 0; i < _poly_order + 1;i++)
	{
		coeff_return.push_back(coeff(i,0));
	}

	return coeff_return;

}

vector<float> LMSFit::FitLine(vector<dwVector2f> _pt_list, uint _poly_order)
{
    mat X = arma::zeros<mat>(_pt_list.size(),1);
    mat Y = arma::zeros<mat>(_pt_list.size(),_poly_order + 1);
    mat coeff = arma::zeros<mat>(_poly_order + 1,1);

    for(uint i = 0; i < _pt_list.size();i++)
    {
        X(i,0) = _pt_list[i].x;

        Y(i,0) = 1;

        for(uint j = 1; j < _poly_order + 1;j++)
        {
            Y(i,j) = pow(_pt_list[i].y,j);
        }
    }

    coeff = arma::solve(Y,X);

    vector<float> coeff_return;

    for(uint i = 0; i < _poly_order + 1;i++)
    {
        coeff_return.push_back(coeff(i,0));
    }

    return coeff_return;
}


bool outlierFilter::RANSACFilter(vector<cv::Point> _raw_ld_result, uint _poly_order, vector<float>& _out_model_coeff, vector<cv::Point>& _out_inlier_pt_list)
{
	srand( (unsigned)time(NULL)+(unsigned)getpid());

    uint RANSAC_N = 100;

	ulong max_inlier_count = 0;

	float inlier_condition = 5;

	vector<cv::Point> pt_list = _raw_ld_result;

	if (pt_list.size() < 10)
		return false;

	for(uint ransac_iter = 0;ransac_iter < RANSAC_N;ransac_iter++)
	{
		uint num_of_picked = 0;
		vector<cv::Point> subset_pt_list;

		ulong count = 0;
		do{
			count++;
			int picked_ind = rand()%pt_list.size();

			cv::Point picked_pt = pt_list[picked_ind];

			bool PickedPtValid = true;

			if(subset_pt_list.size()!=0)
			{
				for(uint check_iter = 0; check_iter < subset_pt_list.size();check_iter++)
				{
					int y_for_aleady_picked = subset_pt_list[check_iter].y;
					int y_current_picked = picked_pt.y;
					if(abs(y_for_aleady_picked-y_current_picked) < 2)
					{
						PickedPtValid = false;
					}
				}

				if(PickedPtValid)
				{
					subset_pt_list.push_back(picked_pt);
					num_of_picked++;
				}
			}
			else
			{
				subset_pt_list.push_back(picked_pt);
				num_of_picked++;
			}
		}while( (num_of_picked < (_poly_order+1)) && (count < pt_list.size()));

		vector<float> cur_model_coeff;

		mat X = arma::zeros<mat>(subset_pt_list.size(),1);
		mat Y = arma::zeros<mat>(subset_pt_list.size(),_poly_order + 1);
		mat coeff = arma::zeros<mat>(_poly_order + 1,1);

		for(uint i = 0; i < subset_pt_list.size();i++)
		{
			X(i,0) = subset_pt_list[i].x;

			Y(i,0) = 1;

			for(uint j = 1; j < _poly_order + 1;j++)
			{
				Y(i,j) = pow(subset_pt_list[i].y,j);
			}
		}

		coeff = arma::solve(Y,X);


		for(uint i = 0; i < _poly_order + 1;i++)
		{
			cur_model_coeff.push_back(coeff(i,0));
		}

		ulong num_of_inliers = 0;
		// Check inlier points for current calculated model

		vector<cv::Point> inlier_pt_list;
		for(uint pt_ind = 0;pt_ind < pt_list.size();pt_ind++)
		{
			int y_val = pt_list[pt_ind].y;
			int x_val = pt_list[pt_ind].x;

			float x_val_est_by_cur_model = 0;
			for(uint order_ind = 0; order_ind < cur_model_coeff.size();order_ind++)
			{
				x_val_est_by_cur_model +=  (float)cur_model_coeff[order_ind]*pow((float)y_val,order_ind);
			}

			if (abs((float)x_val-(float)x_val_est_by_cur_model) <  inlier_condition)
			{
				num_of_inliers++;
				inlier_pt_list.push_back(pt_list[pt_ind]);
			}
		}

		if (num_of_inliers > max_inlier_count)
		{
			max_inlier_count = num_of_inliers;
			_out_model_coeff = cur_model_coeff;
			_out_inlier_pt_list = inlier_pt_list;
		}
	}
	return true;
}

