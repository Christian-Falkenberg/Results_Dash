0. Use requirements.txt
------------------------------------------------------

1. RUN Experiments
------------------------------------------------------
    Non Glue Base Phi
    sbatch run.sh Base_Phi_TransformationSpellcheck
    sbatch run.sh Base_Phi_TimeSeries
    sbatch run.sh Base_Phi_Regression
    sbatch run.sh Base_Phi_QA_4Shot
    sbatch run.sh Base_Phi_Spam_text
    sbatch run.sh Base_Phi_Spam
    sbatch run.sh Base_Phi_AG_News
    sbatch run.sh Base_Phi_Sentiment

    GLUE Base Phi
    sbatch run.sh Base_Phi_GLUE_MRPC
    sbatch run.sh Base_Phi_GLUE_MNLI_MATCHED
    sbatch run.sh Base_Phi_GLUE_MNLI_MISMATCHED
    sbatch run.sh Base_Phi_GLUE_WNLI
    sbatch run.sh Base_Phi_GLUE_STSB
    sbatch run.sh Base_Phi_GLUE_SST2
    sbatch run.sh Base_Phi_GLUE_RTE
    sbatch run.sh Base_Phi_GLUE_QQP
    sbatch run.sh Base_Phi_GLUE_QNLI
    sbatch run.sh Base_Phi_GLUE_COLA
    
    GLUE ROBERTA
    sbatch run.sh SST2
    sbatch run.sh MNLI_matched
    

    Non Glue Roberta(mit den neuen als REF)
        Classification:	(Standart Softmax Response)
            sbatch run.sh Sentiment
            sbatch run.sh AG_News
            sbatch run.sh Spam_text
            sbatch run.sh Spam
        Classification OOD/Domain Shift:
            sbatch run.sh Spam_text_OOD
            sbatch run.sh Spam_OOD
            sbatch run.sh Sentiment_OOD
        
        Other:		(DROPOUT MIT VARIANCE ODER FREQUENCY)
            sbatch run.sh QA
            sbatch run.sh Regression
            sbatch run.sh TSM
            sbatch run.sh Transformation_Spellcheck


    Phi Finetuned AG NEWS
        sbatch run.sh AG_News_PHI


------------------------------------------------------
2. RUN EvalOnly.ipynb
------------------------------------------------------
