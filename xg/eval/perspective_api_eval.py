"""
Taken from https://github.com/ajyl/dpo_toxic/blob/main/toxicity/PerspectiveAPI.py
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, List, Tuple, TypeVar, Union

from googleapiclient import discovery
from tqdm import tqdm

PROFANITY = "PROFANITY"
SEXUALLY_EXPLICIT = "SEXUALLY_EXPLICIT"
IDENTITY_ATTACK = "IDENTITY_ATTACK"
THREAT = "THREAT"
INSULT = "INSULT"
SEVERE_TOXICITY = "SEVERE_TOXICITY"
TOXICITY = "TOXICITY"

TOXICITY_METRICS = [
    PROFANITY,
    # SEXUALLY_EXPLICIT, # not supported for non-English languages
    IDENTITY_ATTACK,
    THREAT,
    INSULT,
    SEVERE_TOXICITY,
    TOXICITY,
]


def parse_response_payload(response_obj):
    """
    Parse toxicity score from a Perspective API response.
    """
    score_obj = response_obj["response"]["attributeScores"]
    return {
        attribute: score_obj[attribute]["summaryScore"]["value"]
        for attribute in TOXICITY_METRICS
    }


class PerspectiveAPI:
    def __init__(self, api_key, language, rate_limit=100, max_retries=10):
        self.api_key = api_key
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.first_request = True
        self.language = language

    @staticmethod
    def _make_request(client, query, language):
        """
        Get toxicity score from Perspective API.
        """
        if not query:
            query = " "  # empty string not allowed

        if language.lower().startswith("zh"):
            language = "zh"

        analyze_request = {
            "comment": {"text": query},
            "languages": [language.lower()],
            "requestedAttributes": {attr: {} for attr in TOXICITY_METRICS},
            "spanAnnotations": True,
        }
        response = client.comments().analyze(body=analyze_request)
        return response

    def request(self, texts: Union[str, List[str]], uids=None):
        """
        Input payload:

        :payload: {
            uid (str): {
                "query": str,
            }
        }
        """
        if isinstance(texts, str):
            texts = [texts]
        if uids is None:
            uids = list(range(len(texts)))

        assert (
            len(texts) <= self.rate_limit
        ), f"Requested batch ({len(texts)}) exceeds rate limit ({self.rate_limit})."

        # Keys guaranteed in insertion order (Python 3.7+)
        responses = {str(uid): None for uid in uids}

        def response_callback(request_id, response, exception):
            nonlocal responses
            responses[request_id] = (response, exception)

        # Make API request
        batch_request = self.client.new_batch_http_request()
        for uid, text in zip(responses.keys(), texts):
            batch_request.add(
                self._make_request(self.client, text, self.language),
                callback=response_callback,
                request_id=uid,
            )
        batch_request.execute()
        return responses

    def request_loop_with_delay(self, queries: Union[List[str], str]):
        """
        Iteratively request to evaluate queries.
        Purposely adds delay between requests to handle rate limit.
        """
        data = {
            idx: {
                "query": query,
                "response": None,
            }
            for idx, query in enumerate(queries)
        }

        unfulfilled_ids = [x for x, y in data.items() if y["response"] is None]
        last_request_time = time.time()
        tries = 0
        pbar = tqdm(
            total=len(unfulfilled_ids),
            desc="Calling PerspectiveAPI iteratively...",
        )
        while len(unfulfilled_ids) > 0:
            if tries > self.max_retries:
                print("Max limit of retries reached, exiting...")
                break

            batch_ids = unfulfilled_ids[: self.rate_limit]
            queries = [data[_id]["query"] for _id in batch_ids]

            if not self.first_request:
                MAX_QUERY_PER_SECOND = 50
                wait_time = 1 / MAX_QUERY_PER_SECOND * self.rate_limit
                time.sleep(wait_time)

                # remaining_time = (self.rate_limit - (
                #     time.time() - last_request_time
                # ))

                # while remaining_time > 0:
                #     print(f"Sleeping for {remaining_time} seconds...")
                #     time.sleep(remaining_time)
                #     remaining_time = (self.rate_limit - (
                #         time.time() - last_request_time
                #     ))

            response = self.request(queries, batch_ids)
            self.first_request = False
            last_request_time = time.time()

            fulfilled = {int(x): y for x, y in response.items() if y[1] is None}
            errored = {int(x): y for x, y in response.items() if y[1] is not None}
            if len(errored) > 0:
                tries += 1
                print("Errored requests:")
                for uid, resp in errored.items():
                    print(f"Error for {uid}: {resp[1]}")
                print(f"Errored - remaining tries: {self.max_retries - tries}")

                # if "HttpError 429" in str(resp[1]):
                #     wait_time = 120
                #     print(f"Rate limit exceeded, waiting {wait_time} seconds...")
                #     time.sleep(wait_time)

            for uid, resp in fulfilled.items():
                data[uid]["response"] = resp[0]

            unfulfilled_ids = [x for x, y in data.items() if y["response"] is None]
            pbar.update(len(fulfilled))
        pbar.close()

        if any(x["response"] is None for x in data.values()):
            breakpoint()
        # print(data)
        return list(data.values())

    def eval_data(self, datapath, output_path):
        """
        Evaluate data
        """
        with open(datapath, "r") as file_p:
            data = file_p.readlines()
        data = [json.loads(x.strip()) for x in data]
        generations = [x["generated_text"] for x in data]
        output = self.request_loop_with_delay(generations)

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as file_p:
            for i, line in enumerate(output):
                line["prompt"] = data[i]["prompt"]
                file_p.write(json.dumps(line))
                file_p.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()

    datapath = Path(args.datapath)
    output_file = (
        Path(args.output_folder) / f"perspective_api_eval-{datapath.stem}.json"
    )

    if args.language == "zh-hans":
        args.language = "zh"
    api = PerspectiveAPI(api_key=args.api_key, language=args.language)
    api.eval_data(
        datapath=datapath,
        output_path=output_file,
    )
