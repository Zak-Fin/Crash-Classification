from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters

class MyPythonModule:
    def hello(self, name):
        return "Hello, " + name

if __name__ == '__main__':
    gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True), callback_server_parameters=CallbackServerParameters())
    gateway.entry_point.register(MyPythonModule())