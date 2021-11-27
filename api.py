from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from modeling import Bert


app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret string'

model = Bert(token_name="bert-base-uncased",
             model_name="test_model",
             train=False)

class Input(FlaskForm):
    query = StringField('Query', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route("/", methods=['POST', 'GET'])
@app.route("/index", methods=['POST', 'GET'])
def predict():
    form = Input()
    if form.validate_on_submit():
        query = form.query.data
        pred_prob, pred_hard = model.predict(query)
        pred_hard = pred_hard[0]
        dic = {"Probability of Michelle quote": str(pred_prob[0][1]),
            "Probability of Barack quote": str(pred_prob[0][0]),
            "Classification": "Barack" if pred_hard == 0 else "Michelle"
            }
        return dic
    else:
        return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True, port=5000)