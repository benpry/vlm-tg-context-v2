"""
Test that we format messages correctly given their histories.
"""

import pandas as pd

from src.utils import preprocess_messages


# TODO: add some tests
def test_preprocess_messages():
    """
    Test that we format messages correctly given their histories.
    """

    # example_row = {
    #     "message_history": [
    #         [
    #             {
    #                 "role": "describer",
    #                 "text": "looks like a person sat down facing the left",
    #                 "message_number": 1,
    #             }
    #         ],
    #         [
    #             {
    #                 "role": "describer",
    #                 "text": "looks like a large person with two pointy bits pointing right",
    #                 "message_number": 1,
    #             },
    #             {
    #                 "role": "describer",
    #                 "text": "rectangular body on left side",
    #                 "message_number": 2,
    #             },
    #         ],
    #     ],
    #     "selection_history": ["A", "B"],
    #     "correctness_history": [True, False],
    #     "message": [
    #         {
    #             "role": "describer",
    #             "text": "looks like a person without legs kind of sat down with their bottom bit pointing right",
    #             "message_number": 1,
    #         },
    #         {"role": "describer", "text": "no arms either", "message_number": 2},
    #     ],
    # }

    # assert preprocess_messages(example_row) == [
    #     {
    #         "role": "user",
    #         "content": "describer: looks like a person sat down facing the left\n",
    #     },
    #     {"role": "assistant", "content": "F"},
    #     {
    #         "role": "user",
    #         "content": "Correct.\n\ndescriber: looks like a large person with two pointy bits pointing right\ndescriber: rectangular body on left side\n",
    #     },
    #     {"role": "assistant", "content": "J"},
    #     {
    #         "role": "user",
    #         "content": "Correct.\n\ndescriber: looks like a person without legs kind of sat down with their bottom bit pointing right\ndescriber: no arms either\n",
    #     },
    # ]

    # test with a real row
    df = pd.read_csv("context_prep/human_history/limited_feedback_yoked.csv")
    assert preprocess_messages(df.iloc[2]) == [
        {
            "role": "user",
            "content": "describer: looks like a person sat down facing the left\n",
        },
        {"role": "assistant", "content": "F"},
        {
            "role": "user",
            "content": "Correct.\n\ndescriber: looks like a large person with two pointy bits pointing right\ndescriber: rectangular body on left side\n",
        },
        {"role": "assistant", "content": "J"},
        {
            "role": "user",
            "content": "Correct.\n\ndescriber: looks like a person without legs kind of sat down with their bottom bit pointing right\ndescriber: no arms either\n",
        },
    ]
