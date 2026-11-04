"""Fold-local vocabulary learning and paired min_df ablation on a fixed holdout."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from .preprocessing import load_task, outer_split, check_split, sha256_file, RESOURCE

MIN_DF_GRID = (1, 2, 3, 5, 10)
MODEL_NAMES = {'logreg':'Логистическая регрессия', 'linear_svm':'Линейный SVM', 'random_forest':'Случайный лес'}

def make_models():
    # Source cell 76 settings; only convergence limits are raised for both arms.
    return {
        'logreg': LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=20000, random_state=42),
        'linear_svm': LinearSVC(C=1.0, class_weight='balanced', dual='auto', max_iter=20000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1),
    }

def make_pipeline(model, min_df):
    return Pipeline([
        ('bow', CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, min_df=min_df)),
        ('model', clone(model)),
    ])

def measure(model, texts, labels):
    pred = model.predict(texts)
    score = model.decision_function(texts) if hasattr(model, 'decision_function') else model.predict_proba(texts)[:,1]
    return {
        'accuracy':accuracy_score(labels,pred),
        'precision_humor':precision_score(labels,pred,zero_division=0),
        'recall_humor':recall_score(labels,pred,zero_division=0),
        'f1_humor':f1_score(labels,pred,zero_division=0),
        'macro_f1':f1_score(labels,pred,average='macro',zero_division=0),
        'roc_auc':roc_auc_score(labels,score),
    }, pred, score

def markdown_table(frame):
    # No optional tabulate dependency.
    def fmt(x):
        return f'{x:.4f}' if isinstance(x, (float,np.floating)) else str(x)
    lines = ['| ' + ' | '.join(map(str,frame.columns)) + ' |', '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)

def compare(train, test, output_dir, provenance=None):
    output_dir = Path(output_dir)
    (output_dir / 'figures').mkdir(parents=True,exist_ok=True)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    check_split(train,test)
    models = make_models()
    inner = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=123)
    folds = list(inner.split(train.text_stem,train.target,groups=train.author_url))
    fold_assignment = np.full(len(train),-1)
    cv_rows = []
    # A failed fit must stop the experiment, not silently produce a partial best score.
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        for fold,(fit_idx,val_idx) in enumerate(folds):
            fit, val = train.iloc[fit_idx], train.iloc[val_idx]
            check_split(fit,val)
            fold_assignment[val_idx] = fold
            print(f'CV fold {fold+1}/5: train={len(fit)}, validation={len(val)}',flush=True)
            for threshold in MIN_DF_GRID:
                for name,template in models.items():
                    pipe = make_pipeline(template,threshold)
                    pipe.fit(fit.text_stem,fit.target)
                    fit_f1 = f1_score(fit.target,pipe.predict(fit.text_stem),average='macro')
                    val_f1 = f1_score(val.target,pipe.predict(val.text_stem),average='macro')
                    cv_rows.append({'model':name,'min_df':threshold,'fold':fold,
                                    'train_macro_f1':fit_f1,'validation_macro_f1':val_f1,
                                    'vocabulary_size':len(pipe['bow'].vocabulary_),
                                    'train_rows':len(fit),'validation_rows':len(val)})
        fold_metrics = pd.DataFrame(cv_rows)
        summary = fold_metrics.groupby(['model','min_df'],as_index=False).agg(
            cv_macro_f1=('validation_macro_f1','mean'),
            cv_std=('validation_macro_f1','std'),
            cv_train_macro_f1=('train_macro_f1','mean'),
            mean_vocabulary_size=('vocabulary_size','mean'))
        summary['cv_train_validation_gap'] = summary.cv_train_macro_f1-summary.cv_macro_f1
        choices = []
        for name in models:
            ranked = summary.loc[summary.model.eq(name)].sort_values(['cv_macro_f1','min_df'],ascending=[False,False],kind='stable')
            filtered = ranked.loc[ranked.min_df.ge(2)].iloc[0]
            choices.append({'model':name,'selected_min_df':int(ranked.iloc[0].min_df),
                            'filtered_min_df':int(filtered.min_df),
                            'selected_cv_macro_f1':float(ranked.iloc[0].cv_macro_f1),
                            'filtered_cv_macro_f1':float(filtered.cv_macro_f1)})
        # Freeze every choice before the first test evaluation.
        selection = pd.DataFrame(choices)
        selection.to_csv(output_dir/'selection.csv',index=False)
        fold_metrics.to_csv(output_dir/'cv_folds.csv',index=False)
        summary.to_csv(output_dir/'cv_summary.csv',index=False)
        metrics_rows, predictions_rows, vocab_rows = [], [], []
        for choice in choices:
            name = choice['model']
            for variant,threshold in [('baseline',1),('filtered',choice['filtered_min_df'])]:
                pipe = make_pipeline(models[name],threshold).fit(train.text_stem,train.target)
                train_metrics,_,_ = measure(pipe,train.text_stem,train.target)
                test_metrics,pred,scores = measure(pipe,test.text_stem,test.target)
                cv = summary.loc[summary.model.eq(name) & summary.min_df.eq(threshold)].iloc[0]
                size = len(pipe['bow'].vocabulary_)
                row = {'model':name,'variant':variant,'min_df':threshold,'vocabulary_size':size,
                       'cv_macro_f1':cv.cv_macro_f1,'cv_std':cv.cv_std,
                       **{'train_'+k:v for k,v in train_metrics.items()},
                       **{'test_'+k:v for k,v in test_metrics.items()},
                       'train_test_gap':train_metrics['macro_f1']-test_metrics['macro_f1']}
                tn,fp,fn,tp = confusion_matrix(test.target,pred,labels=[0,1]).ravel()
                row.update({'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)})
                for split,part in [('train',train),('test',test)]:
                    matrix = pipe['bow'].transform(part.text_stem)
                    row[f'{split}_zero_vectors'] = int((matrix.getnnz(axis=1)==0).sum())
                metrics_rows.append(row)
                for story,author,label,p,s in zip(test.story_id,test.author_url,test.target,pred,scores):
                    predictions_rows.append({'story_hash':hashlib.sha256(str(story).encode()).hexdigest(),
                        'author_hash':hashlib.sha256(str(author).encode()).hexdigest(),
                        'model':name,'variant':variant,'target':int(label),'prediction':int(p),'score':float(s)})
                print(f'{name} {variant}: min_df={threshold}, vocab={size}, test Macro-F1={test_metrics["macro_f1"]:.4f}',flush=True)
        base_vectorizer = CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False)
        base_matrix = base_vectorizer.fit_transform(train.text_stem)
        doc_freq = np.asarray((base_matrix>0).sum(axis=0)).ravel()
        for threshold in MIN_DF_GRID:
            vocab_rows.append({'min_df':threshold,'vocabulary_size':int((doc_freq>=threshold).sum()),
                               'removed_terms':int((doc_freq<threshold).sum()),
                               'removed_fraction':float((doc_freq<threshold).mean())})
        dummy = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train),1)),train.target)
        dummy_metrics,_,_ = measure(dummy,np.zeros((len(test),1)),test.target)
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir/'metrics.csv',index=False)
    pd.DataFrame(predictions_rows).to_csv(output_dir/'test_predictions.csv',index=False)
    vocabulary = pd.DataFrame(vocab_rows)
    vocabulary.to_csv(output_dir/'vocabulary.csv',index=False)
    manifest = []
    for split,part in [('train',train),('test',test)]:
        for i,row in part.iterrows():
            manifest.append({'story_hash':hashlib.sha256(str(row.story_id).encode()).hexdigest(),
                             'author_hash':hashlib.sha256(str(row.author_url).encode()).hexdigest(),
                             'split':split,'target':int(row.target),
                             'inner_validation_fold':int(fold_assignment[i]) if split=='train' else -1})
    pd.DataFrame(manifest).to_csv(output_dir/'split_manifest.csv',index=False)
    run = {'run_utc':datetime.now(timezone.utc).isoformat(),'status':'executed',
           'python':platform.python_version(),
           'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','nltk','matplotlib']},
           'train_rows':len(train),'test_rows':len(test),
           'train_classes':{str(k):int(v) for k,v in train.target.value_counts().sort_index().items()},
           'test_classes':{str(k):int(v) for k,v in test.target.value_counts().sort_index().items()},
           'train_authors':train.author_url.nunique(),'test_authors':test.author_url.nunique(),
           'author_overlap':0,'story_overlap':0,'processed_text_overlap':0,
           'inner_folds':5,'outer_seed':42,'inner_seed':123,'min_df_grid':MIN_DF_GRID,
           'selection_metric':'mean inner validation macro_f1; exact tie -> larger min_df',
           'models':{name:template.get_params() for name,template in models.items()},
           'resource_sha256':sha256_file(RESOURCE),'dummy_test':dummy_metrics,
           'train_singleton_terms':int((doc_freq==1).sum()),'provenance':provenance or {}}
    (output_dir/'run.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    plot_results(metrics,summary,vocabulary,output_dir,len(test))
    write_report(metrics,selection,summary,vocabulary,run,output_dir)
    return {'metrics':metrics,'selection':selection,'cv_summary':summary,'vocabulary':vocabulary,'run':run}

def plot_results(metrics,cv,vocabulary,output_dir,test_rows):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes = plt.subplots(1,3,figsize=(15,4.5))
    for ax,(name,label) in zip(axes,MODEL_NAMES.items()):
        sub = cv.loc[cv.model.eq(name)]
        ax.errorbar(sub.min_df,sub.cv_macro_f1,yerr=sub.cv_std,marker='o',capsize=3,label='Проверка: среднее ± SD')
        ax.plot(sub.min_df,sub.cv_train_macro_f1,'--o',label='Обучение')
        ax.set(title=label,xlabel='min_df (документы)',ylabel='Macro-F1',ylim=(0.35,1.03),xticks=MIN_DF_GRID)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Выбор порога только на train: 5 разбиений по авторам')
    fig.tight_layout()
    fig.savefig(output_dir/'figures/cv_min_df.png',dpi=150)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4.6))
    x = np.arange(3)
    for offset,variant,color in [(-.18,'baseline','#9aa5b1'),(.18,'filtered','#287d8e')]:
        rows = metrics.loc[metrics.variant.eq(variant)].set_index('model').loc[list(MODEL_NAMES)]
        bars = axes[0].bar(x+offset,rows.test_macro_f1,width=.36,label=variant,color=color)
        axes[0].bar_label(bars,fmt='%.3f',padding=3)
    axes[0].set(xticks=x,xticklabels=['Лог. регрессия','Линейный SVM','Случайный лес'],ylabel='Macro-F1 на тесте',ylim=(0,1),title=f'Один и тот же тест: {test_rows} работ')
    axes[0].legend()
    axes[1].plot(vocabulary.min_df,vocabulary.vocabulary_size,marker='o',color='#287d8e')
    axes[1].set(xlabel='min_df (документы)',ylabel='Признаков',title='Словарь полного train',xticks=MIN_DF_GRID)
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir/'figures/comparison.png',dpi=150)
    plt.close(fig)

def write_report(metrics,selection,cv,vocab,run,out):
    shown = metrics[['model','variant','min_df','vocabulary_size','cv_macro_f1','train_macro_f1','test_macro_f1','test_accuracy','test_roc_auc','train_test_gap']].copy()
    deltas = []
    for name,label in MODEL_NAMES.items():
        rows = metrics.loc[metrics.model.eq(name)].set_index('variant')
        a,b = rows.loc['baseline'],rows.loc['filtered']
        deltas.append(f'- **{label}:** min_df={int(b.min_df)}, словарь {int(a.vocabulary_size)} → {int(b.vocabulary_size)}; Macro-F1 теста {a.test_macro_f1:.4f} → {b.test_macro_f1:.4f} (Δ {b.test_macro_f1-a.test_macro_f1:+.4f}); разрыв train−test {a.train_test_gap:.4f} → {b.train_test_gap:.4f}.')
    report = f'''# Результаты сравнения BoW и min_df

Статус: эксперимент выполнен. Полная исходная лабораторная (леммы, TF-IDF, Word2Vec, POS) заново не запускалась.

## Что воспроизведено

{run['train_rows']} работ в train, {run['test_rows']} в test. Классы 0 = Ангст, 1 = Юмор. Авторы, ID и одинаковые обработанные тексты не пересекаются между частями; эти проверки выполнены и внутри CV. Настройки трёх моделей взяты из исходной ячейки 76, которая используется в финальной сравнительной таблице. В первоначальном разделе 1.3 применялся другой подбор параметров — смешивать его показатели с этой таблицей нельзя. Увеличен только предел итераций линейных моделей для сходимости, одинаково в обеих ветках.

## Протокол

Сохранён первый внешний fold StratifiedGroupKFold(5, shuffle=True, random_state=42). Внутри train: 5 групповых folds с seed=123, Pipeline(CountVectorizer, classifier), кандидаты min_df=1,2,3,5,10. На каждом fold словарь обучается заново только по внутреннему train. Фиксированные стоп-слова и исключения стемминга восстановлены из исходного ноутбука, они не обучаются на корпусе. Выбор для каждой модели — максимум среднего Macro-F1 валидации; при точном равенстве выбирается больший порог. SD на графике — разброс по folds, не доверительный интервал.

`selected_min_df` — лучший кандидат, включая 1. `filtered_min_df` — лучший среди 2,3,5,10 для обязательного сравнения фильтрации. Если выбран 1, фильтрация не получила преимущества на CV. Все решения сохранены до оценки теста; параметры классификаторов и граница решения не подбирались по тесту.

{markdown_table(selection)}

## Пересчитанные метрики

{markdown_table(shown)}

Precision, Recall, F1 класса Юмор, матрицы ошибок и число нулевых векторов доступны в `metrics.csv`. Все числа в этой таблице получены текущим запуском, а не перенесены из старых выводов.

{chr(10).join(deltas)}

Базовая модель «всегда Ангст»: Accuracy={run['dummy_test']['accuracy']:.4f}, Macro-F1={run['dummy_test']['macro_f1']:.4f}.

## Как понимать изменение

В исходном словаре {run['train_singleton_terms']} признаков встречаются ровно в одной обучающей работе. `min_df` ограничивает документную частоту: слово, повторённое 50 раз в одном тексте, при min_df=2 всё равно удаляется. После стемминга фильтруются основы/нормализованные токены, а не исходные словоформы.

{markdown_table(vocab)}

Удаление редких признаков уменьшает размерность, но не гарантирует повышения качества: редкие слова могут быть полезны для жанра. Разрыв train/test и train/validation нужно читать вместе с Macro-F1; меньший разрыв при ухудшении теста не является улучшением. При высоком train Macro-F1 нельзя утверждать, что переобучение устранено. У случайного леса также меняется число случайно рассматриваемых признаков, поскольку max_features='sqrt' зависит от размера словаря.

Тест содержит только {run['test_rows']} работ, а его результаты уже присутствовали в исходной лабораторной. Новый подбор его не использует, однако это повторный анализ существующего holdout, а не новый независимый эксперимент. Изменения на этом разбиении не доказывают статистическую значимость. Отсутствие пересечения авторских URL не исключает соавторство, псевдонимы, близкие переписывания или тематическую близость. Метки взяты из карточек работ; отсутствие значка перевода не доказывает оригинальность.

## Воспроизводимость

Текущий словарь baseline содержит 17 342 признака, в сохранённом выводе исходного ноутбука — 17 343. Проверка исходных ячеек на доступном CSV подтвердила совпадение всех 616 обработанных текстов и ID разбиения с перенесённым кодом (`preprocessing_audit.json`). Версии прежней среды и контрольная сумма CSV из старого запуска отсутствуют, поэтому точная причина расхождения не установлена. Новые показатели baseline сравниваются только с новыми filtered, а не со старыми округлёнными метриками.

CSV исходных текстов не включён в публичный архив. Контрольные суммы корпуса и правил, версии библиотек, параметры и размеры выборок находятся в `run.json`. `split_manifest.csv` содержит хеши ID/авторов, метки и принадлежность к fold; `test_predictions.csv` — предсказания без текстов. `source_audit.json` хранит отдельно исторические округлённые числа.

![Подбор min_df](figures/cv_min_df.png)

![Сравнение](figures/comparison.png)

Документация: [CountVectorizer: min_df](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), [предотвращение утечек](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
'''
    (out/'results.md').write_text(report,encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('ficbook_all_collected_1900.csv'))
    parser.add_argument('--output',type=Path,default=Path('reports'))
    args = parser.parse_args()
    task,counts = load_task(args.data)
    train,test = outer_split(task)
    compare(train,test,args.output,{'dataset_name':args.data.name,'dataset_sha256':sha256_file(args.data),**counts})

if __name__ == '__main__':
    main()
