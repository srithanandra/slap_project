import ai_controller

MODEL_PATH = 'src//model_5_13.pt'

def main():
    controller = ai_controller.ModelController(MODEL_PATH)
    controller.run_loop(frequency=50)

if __name__ == '__main__':
    main()