"""
@author:Sagar Sinha

"""
from flask import Flask, render_template, request
from flask_cors import cross_origin
import joblib 



#Initialising the app and loading the models
app = Flask(__name__)

xgbr_model = joblib.load("./saved models/xgbr.pkl")
lgbm_model = joblib.load("./saved models/lgbm.pkl")
rmr_model = joblib.load("./saved models/rmr.pkl")
print("Models Loaded")
    
#Use flask_cors or secret key



@app.route("/", methods=['GET']) #This signifies the default route
@cross_origin()
def home():
    return render_template("index.html", title="Home Page")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        
         # Monthly EMI
         emi = float(request.form['EMI'])
         
         # Income
         income = float(request.form['Income'])
        
		 # Loan to Value Ratio
         ltv = float(request.form['LTV'])
         
		 # Asset Cost
         assetCost = float(request.form['Asset-Cost'])
         
		 # Age
         age = float(request.form['Age'])
         
		 # Mean Balance
         meanBal = float(request.form['Balance-Mean'])
        
		 # Maximum Current Balance
         currBal = float(request.form['Current-Bal'])
        
		 # Disbursed Amount
         disbursedAmt = float(request.form['Disbused-Amt'])
        
		 # Finance-Amount
         amtFin = float(request.form['Amt-Fin'])
         
		 # Minimum Current Balance Sum
         minCBS = float(request.form['Min-CBS'])
         
		 # Delays in request
         delays = float(request.form['Delays'])
        
		 # Count of Payments Past Due Date
         paymentPD = float(request.form['Payment-PP'])
        
		 # Mean of Successive Current Balance Differences
         meanSCB = float(request.form['Mean-SCB'])

		 # Sum of number of delays in payment
         sumDelays = float(request.form['Delays-Sum'])
         
		 # Minimum Current Balance
         minCB = float(request.form['Min-CB'])
        
		 # Mean of month count with no history of prior payment
         meanMC = float(request.form['MC'])
        
		 # Maximum Current Balance
         maxCB = float(request.form['MaxCB'])
        
		 # Mean of Maximum Current Balances for no recorded Payment History
         hout = float(request.form['HOUT'])
        
		 # Maximum of Minimum of Current Balances
         mamiCB = float(request.form['MaMiCB'])
        
		 # Current Balance
         cb = float(request.form['CB'])
        
		 # Tenure
         tenure = float(request.form['Tenure'])
        
		 # Mean of Sum of Successive Current Balance Differences
         meanBD = float(request.form['Mean-BD'])
        
         # Sum of month count with no history of prior payment
         mchprp = float(request.form['MCHPRP'])
        
		 # Maximum Count of Payment Past Due Date
         mcppdd = float(request.form['MCPPDD'])
        
		 # Maximum Count of Number of delays in Payment
         mcnpaym = float(request.form['MCNPAYM'])
        
		 # 0 or 1 as per the history of outstanding payments
         hhopaym = float(request.form['HHOPAYM'])
        
		 # Mean of Maximum Amount Paid
         mmaapa= float(request.form['MMaAPA'])
        
		 # Mean of Sum of Amount Paid
         mosoap = float(request.form['MOSOAP'])
        
		 # Maximum of Maximum Amount Paid
         mamaap= float(request.form['MaMaAP'])
        
		 # 0 or 1 as per the history of outstanding payments
         hhopay = float(request.form['HHOPAY'])
         
         # Maximum of Sum of Amount Paid
         masap = float(request.form['MASAP'])
         
         # Minimum Count of No. of Delays in Payment
         mcndp = float(request.form['MCNDP'])
         
         # Maximum of Average Successive Difference in Current Balance
         masdcb = float(request.form['MASDCB'])
         
         # Minimum Current Balance
         minicb = float(request.form['MiniCB'])
         
         # Minimum Current Balance
         minicb = float(request.form['MSSDCB'])
         
         # 0 or 1 as per the history of outstanding payments
         hhopa = float(request.form['HHOPA'])
         
         # Minimum Count of Payments Past Due Date
         mmcpppd = float(request.form['MMCPPPD'])
         
         # Minimum Count of Payments Past Due Date
         memcb = float(request.form['MeMCB'])
         
         # Minimum of Minimum Current Balance
         mmcb = float(request.form['MMCB'])
         
         # 0 or 1 as per the history of outstanding payments
         o1h0 = float(request.form['01HO'])
         
         # Sum of Minimum Current Difference in Balance
         smcb = float(request.form['SMCB'])
         
         # Minimum of Minimum Amount Paid
         mimap = float(request.form['MiMAP'])
         
         # Sum of Minimum Amount Paid
         smap = float(request.form['SMAP'])
         
         # Maximum of Minimum Amount Paid
         mmap = float(request.form['MMAP'])

         input_lst = [emi, income, ltv, assetCost, age, meanBal, currBal, disbursedAmt, 
                     amtFin, minCBS, delays, paymentPD, meanSCB, sumDelays, minCB, meanMC, maxCB, 
                     hout, mamiCB, cb, tenure, meanBD, mchprp, mcppdd, mcnpaym, hhopaym,
                     mmaapa, mosoap, mamaap, hhopay, masap, mcndp, masdcb, minicb, hhopa, 
                     masap, mcndp, masdcb, minicb, hhopa, mmcpppd, memcb, mmcb, o1h0, smcb, mimap, 
                     smap, mmap]
        
         pred = lgbm_model.predict(input_lst)
         output = pred
         print(output)
    
    return render_template("predictor.html", title="Prediction Page")

#Run the app from main method
if __name__ == '__main__':
   app.run(debug=True)