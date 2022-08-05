# KFs_TSDL
Title: Kernel Flows for learning dynamical systems from data with application to time series forecasting

Learning dynamical systems from data can be done by regressing the vector field with a
kernel. In this project, we present kernel flow algorithm as a numerical approach to learn the
kernel from data. We explore the performance of this algorithm in the context of time series
forecasting from the benchmarking dataset TSDL.

-->"fit_rkhs_stdl.py"           : contains the code of Kernel Flows and the loss functions.
-->"test_dynamical_systems.py"  : in this file, we can apply KFs to some chaotic dynamical systems 
 (Logistic map, Hénon map and Lorenz system) and to a synthetic time series.
-->"test_henon_attractor.py"    : to reconstruct the henon attractor using KFs. 
-->"test_lorenz_attractor.py"   : to reconstruct the lorenz attractor using KFs. 
-->"main_map.py"                : to forecast time series from the TSDL library.
-->tsdl.JSON                    : contains a set of 648 time series from the TSDL library.

