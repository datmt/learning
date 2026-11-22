from time import sleep

class Dummy():
    
    def __init__(self, delay):
        self.delay = delay

    def get_name(self):
        return "Dummy"
    def load(self):
        sleep(0.5)

    def predict(self, request):
        sleep(self.delay)

        return {
            "output": request,
            "metrics": {}
        }
    def unload(self):
        pass