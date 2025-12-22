from controller_agent import VisionLLMAgent

def main():
    agent = VisionLLMAgent()

    print("\n👁️ LLM with Eyes is running")
    print("Type a question like: 'What is in my hand?'")
    print("Press Ctrl+C to exit\n")

    while True:
        try:
            question = input("🧑 You: ")
            answer = agent.run(question)
            print(f"\n🤖 Agent: {answer}\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
