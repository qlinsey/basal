PROJECT: B-ASAL active learning framework

1. Install
	pip install -r requirements.txt

2. Framework

	- Multi-Class Discriminator/Classifier 
	- Bi-Discriminator 
	- Feature Encoder 

3. Workflow
        - Cold-start data points generation -> Human Label -> Retrain Model -> Need more label (Y/N) ->Y, Retrain Model->...-> converge 


4. Components

	- Dataset 
    - Cold-start Samples
	- Training
	- Performance report 
	- Inference/Score 
	- Query for labelling 
	- Add labelled data 
	- Generate pesudo labelled data

4. Execution 

	- Tree 
	  sh run_main.sh tree

	- Data info:
	  sh run_main.sh info
	
	- Run Model Training:
	  sh run_main.sh train
	 
	- Collecting performance report:
	  sh run_main.sh perf 

	- Query data to be labelled:
	  sh run_main.sh query

	- Predict/Score data points:
	  sh run_main.sh score

    - Retrain or not:
          sh run_main.sh retrain
    - Cold Start:
          sh run_main.sh coldstart


5. References
    
