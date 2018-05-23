from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from json import dumps

app = Flask(__name__)
api = Api(app)


testdict = {"test1": "hello1", "test2": "hello2"}
managedict = {'key1': 'val1'}

class Employees(Resource):
    def get(self):  
        return managedict  # Fetches first column that is Employee ID


class test11(Resource):
    def get(self):
        return "This test worked!!!!2"

    def post(self):
        parser = reqparse.RequestParser()
        args = parser.parse_args()
        print (args)
        return "Ok"

class Employees2(Resource):
    def get(self, keyToAdd, valueToAdd):
        managedict[keyToAdd] = valueToAdd
        return "Added the following to the 'Manage' dictionary:{0}: {1}".format(keyToAdd, valueToAdd)


class testdict1(Resource):
    def get(self, test_id):
        return "{testid}: {dictitem}".format(testid=test_id, dictitem=testdict[test_id])
    def post(self):
        value = "what"
        return value

class value(Resource):
   # @app.route('/post/', methods=['POST'])
    def post(self):
        #json_data = request.get_json(force=True)
        #print(json_data)
        data = request.stream
        print(data)
        return "data"

class dictmanage(Resource):
    def get(self):
        return managedict


api.add_resource(test11, '/test11')
api.add_resource(testdict1, '/testdict1/<string:test_id>')
#api.add_resource(testdict1, '/testdict1')
api.add_resource(value, '/value')


api.add_resource(dictmanage, '/manage')
api.add_resource(Employees, '/employees') # Route_1

api.add_resource(
    Employees2, '/manageadd/<string:keyToAdd>/<string:valueToAdd>')

if __name__ == '__main__':
     app.run(port=5002)
