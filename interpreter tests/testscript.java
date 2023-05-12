import py4j.GatewayServer;

public class MyJavaProgram {
    public static void main(String[] args) {
        GatewayServer gateway = new GatewayServer(new MyPythonModule());
        gateway.start();

        MyPythonModule module = gateway.getEntryPoint();
        String greeting = module.hello("Fin");
        System.out.println(greeting);

        gateway.shutdown();
    }
}