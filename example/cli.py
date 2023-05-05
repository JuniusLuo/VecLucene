
import argparse
import requests
import sys
import time

def upload_file(url: str, file_path: str):
    with open(file_path, 'rb') as f:
        resp = requests.post(
            url=url, files={'file': (f.name, f, "text/plain")})
        print(resp.json())


def upload_file_with_fields(url: str, file_path: str):
    with open(file_path, 'rb') as f:
        field1 = '{"name": "field1", "string_value": "str1"}'
        field2 = '{"name": "field2", "numeric_value": 2}'
        doc_fields = '{"fields": ' + f'[{field1}, {field2}]' + '}'
        fields = {"fields": f'{doc_fields}'}
        resp = requests.post(
            url=url, files={'file': (f.name, f, "text/plain")}, data=fields)
        print(resp.json())


def commit(url: str):
    resp = requests.post(url=url)
    print(resp.json())


def query(url: str, query_string: str, query_type: str):
    query_request = '{' + f'"query": "{query_string}", ' + \
                    f'"query_type": "{query_type}"' + '}'
    print(query_request)
    resp = requests.get(url=url, data=query_request)
    print(resp.json())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--file", type=str)
    parser.add_argument("--query_string", type=str)
    parser.add_argument("--query_type", type=str, default="vector")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    match args.op:
        case "upload":
            if args.file is None:
                print("please input the text file path")
            url += "/add_doc"
            upload_file(url, args.file)

        case "commit":
            url += "/commit"
            commit(url)

        case "query":
            if args.query_string is None:
                print("please input the query string")
            url += "/query"
            start = time.monotonic()
            query(url, args.query_string, args.query_type)
            dur = time.monotonic() - start
            print(f"{args.query_type} query time: {dur}s")

        case _:
            print("supported op: upload, commit, query")

