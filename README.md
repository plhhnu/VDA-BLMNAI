This code is written by:

https://github.com/stephenliu0423/PyDTI

and is modified by XiongFei Tian.

Email: txfhut@163.com

For any questions regarding to this library, please feel free to contact the author.

parameters：

--method 			set VDA prediction method
--dataset: 			choose the benchmark dataset, i.e., vd
--folder:			set the the folder that contains the datasets 
--csv:				choose the cross-validation setting, 1 for CVS1, 2 for CVS2, and 3 for CVS3, (default 1)
--specify-arg:		0 for choosing optimal arguments, 1 for using default/specified arguments (default 1)
--method-opt:		set arguments for each method (method ARGUMENTS have the form name=value)
--predict-num:		0 for not predicting novel DTIs, a positive integer for predicting top-N novel DTIs (default 0)

Here are some examples:

run a method with default arguments:
	python VDI.py --method="nrlmf" --dataset="nr"
	python VDI.py --method="nrlmf" --dataset="nr" --cvs=2
	python VDI.py --method="nrlmf" --dataset="nr" --cvs=2 --specify-arg=1
