import json
from scipy.special import expit
from random import sample, shuffle
from statistics import mean
import numpy as np
import pandas as pd
import pickle

DEFAULT_PARAMETERS = (-0.06760512, 0.1, -0.01688316, -0.21032119, 0.00128378, -0.02590103, -0.30148615, 0.03580852, -0.0356914)

with open('EEE-196/Compiler/aicomprehend_annotated_dataset_v7.json') as file:
    MASTER_DATA = json.load(file)


class StudentModel:
    def __init__(self, student_id, student_history, student_parameters=DEFAULT_PARAMETERS, remaining_question_ids=None, diagnostic_test_ids=None,
                mastered_components=None, inappropriate_components=None, model=None, in_diagnostic=False, in_review=False):

        self.student_id = student_id
        self.student_history = student_history
        self.recent_history = student_history[-30:]
        # # get knowledge component of last 30 questions using student_history and MASTER_DATA
        # self.recent_history = [MASTER_DATA[i['question_id']]['knowledge_component'] for i in self.recent_history]

        # get correct/incorrect of last 30 questions using student_history
        self.correct_responses = {
            'literal': [i['correct'] for i in self.recent_history if MASTER_DATA[i['question_id']]['knowledge_component'] == 'literal'].count(True),
            'inferential': [i['correct'] for i in self.recent_history if MASTER_DATA[i['question_id']]['knowledge_component'] == 'inferential'].count(True),
            'critical': [i['correct'] for i in self.recent_history if MASTER_DATA[i['question_id']]['knowledge_component'] == 'critical'].count(True)}

        self.incorrect_responses = {
            'literal': [i['correct'] for i in self.recent_history if MASTER_DATA[i['question_id']]['knowledge_component'] == 'literal'].count(False),
            'inferential': [i['correct'] for i in self.recent_history if MASTER_DATA[i['question_id']]['knowledge_component'] == 'inferential'].count(False),
            'critical': [i['correct'] for i in self.recent_history if MASTER_DATA[i['question_id']]['knowledge_component'] == 'critical'].count(False)}
        self.correct_responses['literal'] += self.correct_responses['inferential'] + self.correct_responses['critical']
        self.correct_responses['inferential'] += self.correct_responses['critical']
        self.incorrect_responses['literal'] += self.incorrect_responses['inferential'] + self.incorrect_responses['critical']
        self.incorrect_responses['inferential'] += self.incorrect_responses['critical']
        self.student_parameters = student_parameters

        if diagnostic_test_ids is None:
            # add 3 unique questions from each knowledge component
            self.diagnostic_ids = []
            self.diagnostic_ids.extend(sample([i['id'] for i in MASTER_DATA if i['knowledge_component'] == 'literal'], 3))
            self.diagnostic_ids.extend(sample([i['id'] for i in MASTER_DATA if i['knowledge_component'] == 'inferential'], 3))
            self.diagnostic_ids.extend(sample([i['id'] for i in MASTER_DATA if i['knowledge_component'] == 'critical'], 3))
            self.remaining_question_ids = [i['id'] for i in MASTER_DATA if i['id'] not in self.diagnostic_ids]
        else:
            self.diagnostic_ids = diagnostic_test_ids

        if mastered_components is None:
            self.mastered_components = []
        else:
            self.mastered_components = mastered_components
        if inappropriate_components is None:
            self.inappropriate_components = []
        else:
            self.inappropriate_components = inappropriate_components

        if model is None:
            self.model = self.model_chooser()
        else:
            self.model = model

        self.in_review = in_review
        self.in_diagnostic = in_diagnostic

    def model_chooser(self):
        # if student_id is odd, use model 1, if even, use model 2
        if self.student_id % 2 == 0:
            return '1'
        else:
            return '2'

    def pfa_model(self):

        beta_literal, gamma_literal, rho_literal, beta_inferential, gamma_inferential, rho_inferential, beta_critical, \
            gamma_critical, rho_critical = self.student_parameters

        m_literal = beta_literal + gamma_literal * self.correct_responses['literal'] + rho_literal * \
                    self.incorrect_responses['literal']
        m_inferential = beta_inferential + gamma_inferential * self.correct_responses['inferential'] + rho_inferential * \
                        self.incorrect_responses['inferential']
        m_critical = beta_critical + gamma_critical * self.correct_responses['critical'] + rho_critical * \
                     self.incorrect_responses['critical']

        p_literal = expit(m_literal)
        p_inferential = expit(m_inferential + m_literal)
        p_critical = expit(m_critical + m_inferential + m_literal)

        prediction = {'literal': np.clip(p_literal, 0.01, 0.99), 'inferential': np.clip(p_inferential, 0.01, 0.99),
                      'critical': np.clip(p_critical, 0.01, 0.99)}

        self.mastered_components = [i for i in prediction if prediction[i] > 0.8 and i not in self.mastered_components]
        self.inappropriate_components = [i for i in prediction if prediction[
            i] < 0.2 and i != 'literal' and i not in self.inappropriate_components and i not in self.mastered_components]

        return prediction

    def log_res_vanilla(self):
        with open('log_res.pkl', 'rb') as model_file:
            log_res = pickle.load(model_file)

        data = pd.DataFrame({'kc_literal', 'kc_inferential', 'kc_critical',
                             'kc_literal_success', 'kc_inferential_success', 'kc_critical_success',
                             'kc_literal_failure', 'kc_inferential_failure', 'kc_critical_failure'})

        data.loc[0] = [0, 0, 0, self.correct_responses['literal'], self.correct_responses['inferential'],
                        self.correct_responses['critical'], self.incorrect_responses['literal'],
                        self.incorrect_responses['inferential'], self.incorrect_responses['critical']]

        literal_data = inferential_data = critical_data = data.copy()
        literal_data['kc_literal'] = 1
        literal_data['kc_inferential'] = literal_data['kc_critical'] = 0

        inferential_data['kc_literal'] = inferential_data['kc_inferential'] = 1
        inferential_data['kc_critical'] = 0

        critical_data['kc_literal'] = critical_data['kc_inferential'] = critical_data['kc_critical'] = 1

        prediction = {'literal': np.clip(log_res.predict_proba(literal_data)[0][1], 0.01, 0.99),
                      'inferential': np.clip(log_res.predict_proba(inferential_data)[0][1], 0.01, 0.99),
                      'critical': np.clip(log_res.predict_proba(critical_data)[0][1], 0.01, 0.99)}

        self.mastered_components = [i for i in prediction if prediction[i] > 0.8 and i not in self.mastered_components]
        self.inappropriate_components = [i for i in prediction if prediction[
            i] < 0.2 and i != 'literal' and i not in self.inappropriate_components and i not in self.mastered_components]

        return prediction

    def model_response(self):

        if len(self.recent_history) == 0:
            self.in_diagnostic = True

        if self.in_diagnostic and len(self.recent_history) == 9:
            self.in_diagnostic = False
            next_question_id = self.diagnostic_ids[len(self.diagnostic_ids)]
            shuffle(self.diagnostic_ids)
        elif self.in_diagnostic and len([i for i in self.student_history if i['question_id'] in self.diagnostic_ids]) == 18:
            self.in_diagnostic = False
            self.in_review = True
            self.remaining_question_ids = [i['id'] for i in MASTER_DATA]
            next_question_id = sample(self.remaining_question_ids, 1)
        elif self.in_diagnostic and len(self.mastered_components) == 3:
            next_question_id = self.diagnostic_ids[len(self.recent_history)]
        elif self.in_diagnostic:
            next_question_id = self.diagnostic_ids[len(self.recent_history)]
        elif self.in_review:
            next_question_id = sample(self.remaining_question_ids, 1)
        else:
            if self.model == '1':
                prediction = self.pfa_model()
            else:
                prediction = self.log_res_vanilla()

            expectation = {}
            for i in prediction:
                expectation[i] = prediction[i] * mean(prediction.values()) + (1 - prediction[i]) * (
                        1 - mean(prediction.values()))

            # remove mastered components and inappropriate components from expectation
            expectation = {i: expectation[i] for i in expectation if i not in self.mastered_components and i not in self.inappropriate_components}

            next_question_kc = max(expectation, key=expectation.get)

            # get a random question id from the remaining questions ids
            next_question_id = sample([i['id'] for i in MASTER_DATA if i['knowledge_component'] == next_question_kc] and i['id'] not in self.remaining_question_ids, 1)

        return next_question_id, self.mastered_components, self.remaining_question_ids, self.diagnostic_ids, \
               self.inappropriate_components, self.model, self.in_review, self.in_diagnostic
