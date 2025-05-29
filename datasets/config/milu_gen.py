from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MILUDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

# List of available languages in MILU
milu_languages = [
    'Bengali',
    'English',
    'Gujarati',
    'Hindi',
    'Kannada',
    'Malayalam',
    'Marathi',
    'Odia',
    'Punjabi',
    'Tamil',
    'Telugu'
]

milu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev')

milu_datasets = []
for language in milu_languages:
    _hint = f'There is a single choice question in {language}. Answer the question by replying A, B, C or D.'
    milu_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
                ),
                dict(role='BOT', prompt='{target}\n')
            ]),
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=
                        f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
                    ),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer),
    )

    milu_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

    milu_datasets.append(
        dict(
            abbr=f'ai4bharat_milu_{language}',
            type=MILUDataset,
            path='ai4bharat/MILU',
            language=language,
            reader_cfg=milu_reader_cfg,
            infer_cfg=milu_infer_cfg,
            eval_cfg=milu_eval_cfg,
        ))

del language, _hint