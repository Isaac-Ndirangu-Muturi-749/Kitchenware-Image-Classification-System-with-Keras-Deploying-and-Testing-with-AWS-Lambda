import requests

def invoke_lambda_function(image_url):
    # deploy by running docker image locally on port 9000
    # lambda_invoke_url = 'http://localhost:9000/2015-03-31/functions/function/invocations'
    # publish docker image to aws ecr, creating aws lambda function & exposing it via aws API gateway
    lambda_invoke_url = 'https://9vmhopr686.execute-api.eu-west-3.amazonaws.com/testAPI/predict'
    payload = {'url': image_url}

    response = requests.post(lambda_invoke_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__":
    test_image_url = 'https://th.bing.com/th/id/OIP.LdDqEwKlL3wJ1zFPjzvFqwHaIW?rs=1&pid=ImgDetMain'
    result = invoke_lambda_function(test_image_url)
    print('image 1')
    print(result)
    print()


    test_image_url = 'https://th.bing.com/th/id/R.72f44a5975ee79e83703cfb10809f0be?rik=6ht3ChPx7G84yA&pid=ImgRaw&r=0'
    result = invoke_lambda_function(test_image_url)
    print('image 2')
    print(result)