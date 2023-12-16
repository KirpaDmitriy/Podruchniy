import json
import os

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd


def get_answer_gpt(message):
    openai.api_key = os.environ.get("OPENAI_KEY")
    messages = [{"role": "system", "content": "You are an intelligent assistant."}]
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        reply = chat.choices[0].message.content
        return reply


def delete_unnecessary_column(df):
    delete_column = []
    for column in df.columns:
        if len(df[column].unique()) == 1 or len(df[column].unique()) == len(df):
            delete_column.append(column)
    df = df.drop(columns=delete_column)
    return df, delete_column


def column_conversion(df):
    df_copy = df.copy()
    df_copy = df_copy.fillna(0)
    for column in df_copy.columns:
        if df_copy[column].dtype == "datetime64[ns]":
            df_copy[column] = df_copy[column].astype("category")
            df_copy[column] = df_copy[column].cat.codes
        if df_copy[column].dtype == "object":
            df_copy[column] = df_copy[column].astype("category")
            df_copy[column] = df_copy[column].cat.codes
    return df_copy


def get_all_corr(all_data_copy):
    # вывести все возможные корреляции
    df_corr = pd.DataFrame(columns=["column1", "column2", "corr", "abs_corr"])
    for i in range(len(all_data_copy.columns)):
        for j in range(0, i):
            if (
                i != j
                and abs(
                    all_data_copy[all_data_copy.columns[i]].corr(
                        all_data_copy[all_data_copy.columns[j]]
                    )
                )
                != 0
            ):
                df_corr.loc[len(df_corr)] = [
                    all_data_copy.columns[i],
                    all_data_copy.columns[j],
                    all_data_copy[all_data_copy.columns[i]].corr(
                        all_data_copy[all_data_copy.columns[j]]
                    ),
                    abs(
                        all_data_copy[all_data_copy.columns[i]].corr(
                            all_data_copy[all_data_copy.columns[j]]
                        )
                    ),
                ]

    df_without_one_unique = df_corr.copy()
    for col1, col2 in df_corr[["column1", "column2"]].values:
        if (
            len(all_data_copy[col1].value_counts()) == 1
            or len(all_data_copy[col2].value_counts()) == 1
        ):
            df_without_one_unique = df_without_one_unique[
                ~(
                    (df_without_one_unique["column1"] == col1)
                    & (df_without_one_unique["column2"] == col2)
                )
            ]
    df_without_one_unique = df_without_one_unique.sort_values(
        by=["abs_corr"], ascending=False
    )
    df_corr = df_corr.sort_values(by=["abs_corr"], ascending=False)
    return df_without_one_unique, df_corr


def graph_with_corr_data(df, col1, col2, info_data, column_desc):
    df_to_graph = df[[col1, col2]]
    df_to_graph = df_to_graph[
        ~((df_to_graph[col1].isna()) | (df_to_graph[col2].isna()))
    ]

    type_graph = "linear"

    if df[col1].dtype == object and df[col2].dtype == object:
        type_graph = "circle"

    elif df[col1].dtype == object or df[col2].dtype == object:
        type_graph = "bar"

    elif (
        df[col1].dtype == np.int64
        and len(df[col1].unique()) <= 20
        or df[col2].dtype == np.int64
        and len(df[col2].unique()) <= 20
    ):
        type_graph = "bar"

    df_to_graph["type"] = type_graph
    return df_to_graph


def graph_one_column(df, col1):
    df = df[~df[col1].isna()]
    if (df[col1].dtype == np.float64 or df[col1].dtype == np.int64) and len(
        df[col1].unique()
    ) > 20:
        n, bins, _ = plt.hist(df[col1], color="blue", edgecolor="black", bins=10)
        return pd.DataFrame(data=zip(bins, n), columns=["value", "count"])
    return df[col1].value_counts().reset_index().rename(columns={col1: "value"})


def return_for_front(all_data, col1, col2, corr, info_data, column_desc):
    message_for_gpt = f'У нас есть {info_data}. Сформулируй гипотезы о данных, зная, что корреляция между колонками "{column_desc[col1]}" и "{column_desc[col2]}" составляет {round(corr,2)}. Сформулируй ответ в трёх обзацах: в первом очень кратко и просто в двух-трёх словах расскажи суть гипотезы, во втором - распиши очень подробно, в третьем - предложи способы проверки гипотезы'
    answer_gpt = get_answer_gpt(message_for_gpt)

    dict_result = dict()
    dict_result["message"] = answer_gpt
    dict_result["corr"] = corr
    df_to_graph = graph_with_corr_data(all_data, col1, col2, info_data, column_desc)
    df_to_graph = df_to_graph.reset_index(drop=True)
    if df_to_graph["type"][0] != "linear":
        df_len = len(df_to_graph[col1])
        print("Data init len: %s", df_len)
        df_step = df_len // 1_000
        df = pd.crosstab(df_to_graph[col1], df_to_graph[col2]).loc[::df_step]
        print("Data init len: %s", len(df))
        dict_result["first_graph"] = {
            "name": f"График зависимости переменной {col1} от {col2}",
            "type": df_to_graph["type"][0],
            "data": df.to_json(),
        }
    else:
        df_len = len(df_to_graph[[col1, col2]])
        print("Data init len: %s", df_len)
        df_step = df_len // 1_000
        df_to_graph1 = df_to_graph[[col1, col2]].loc[::df_step]
        print("Data init len: %s", len(df_to_graph1))
        dict_result["first_graph"] = {
            "name": f"График зависимости переменной {col1} от {col2}",
            "type": df_to_graph["type"][0],
            "data": df_to_graph1.to_json(),
        }
    dict_result["second_graph"] = {
        "name": f"График распределения {col1}",
        "type": "hist",
        "data": graph_one_column(all_data, col1).to_json(),
    }
    dict_result["thirs_graph"] = {
        "name": f"График распределения {col2}",
        "type": "hist",
        "data": graph_one_column(all_data, col2).to_json(),
    }
    return dict_result


def get_description(all_data):
    message_for_gpt = f"Есть данные про колонки: {all_data.columns}. Выведи в виде json-словаря название из списка и описание к нему"
    answer = get_answer_gpt(message_for_gpt).replace("'", '"')
    try:
        answer = json.loads(answer)
        return answer
    except Exception:
        raise Exception(f"Broken: {answer}")


def get_info(description):
    message_for_gpt = f"Есть данные и описание к ним:{description} Напиши в 3-5 словах, о чем эти данные"
    return get_answer_gpt(message_for_gpt)


def get_hypotheses(all_data: pd.DataFrame) -> list[dict]:
    all_data = column_conversion(delete_unnecessary_column(all_data)[0])
    all_data_copy = column_conversion(delete_unnecessary_column(all_data)[0])
    df_without_one_unique, _ = get_all_corr(all_data_copy)
    column_desc = get_description(all_data)
    info_data = get_info(column_desc)
    result = []
    corr_col_data = df_without_one_unique[df_without_one_unique.abs_corr >= 0.75][
        ["column1", "column2", "corr"]
    ].values
    i = 1
    for col1, col2, corr in corr_col_data[:3]:
        print(f"Begin processing: {i}")
        result.append(
            return_for_front(all_data, col1, col2, corr, info_data, column_desc)
        )
        print(f"Begin processing: {i}")
        i += 1
    return result