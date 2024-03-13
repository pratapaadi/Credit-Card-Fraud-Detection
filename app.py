from flask import Flask,render_template,request,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
app=Flask(__name__)

@app.route('/')
def home_page():
 return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Time=float(request.form.get('Time')),
            Amount= float(request.form.get('Amount')),
            TransactionMethod = float(request.form.get('TransactionMethod')),
            TransactionId= float(request.form.get('TransactionId')),
            Location= float(request.form.get('Location')),
            TypeofCard= float(request.form.get('TypeofCard')),
            Bank= request.form.get('Bank'),
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        if pred==0:
           results="Transaction is not fraudlent."
        if pred==1:
           results="Transaction is fraudlent."
        return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
