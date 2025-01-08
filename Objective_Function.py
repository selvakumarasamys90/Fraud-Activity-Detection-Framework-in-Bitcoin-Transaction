import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_vars
from Model_Proposed import Model_PROPOSED


def Objective_Function(Soln):
    Feat = Global_vars.Feat
    Tar = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, predict = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(predict, Test_Target)
            Fitn[i] = 1 / (Eval[4] + Eval[7] + Eval[13]) + Eval[11]
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, predict = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(predict, Test_Target)
        Fitn = 1 / (Eval[4] + Eval[7] + Eval[13]) + Eval[11]
        return Fitn