###############
# AGENT MODEL #
###############
import datetime
from besser.BUML.metamodel.state_machine.state_machine import Body, Condition, ConfigProperty, CustomCodeAction
from besser.BUML.metamodel.state_machine.agent import Agent, AgentReply, LLMReply, RAGReply, DBReply, LLMOpenAI, LLMHuggingFace, LLMHuggingFaceAPI, LLMReplicate, RAGVectorStore, RAGTextSplitter, ReceiveTextEvent, ReceiveFileEvent, ReceiveJSONEvent, ReceiveMessageEvent, WildcardEvent, DummyEvent
from besser.BUML.metamodel.structural import Metadata
import operator

agent = Agent('Gym_Agent')

agent.add_property(ConfigProperty('websocket_platform', 'websocket.host', '0.0.0.0'))
agent.add_property(ConfigProperty('websocket_platform', 'websocket.port', 8765))
agent.add_property(ConfigProperty('websocket_platform', 'streamlit.host', '0.0.0.0'))
agent.add_property(ConfigProperty('websocket_platform', 'streamlit.port', 5000))
agent.add_property(ConfigProperty('nlp', 'nlp.language', 'en'))
agent.add_property(ConfigProperty('nlp', 'nlp.region', 'US'))
agent.add_property(ConfigProperty('nlp', 'nlp.timezone', 'Europe/Madrid'))
agent.add_property(ConfigProperty('nlp', 'nlp.pre_processing', True))
agent.add_property(ConfigProperty('nlp', 'nlp.intent_threshold', 0.4))
agent.add_property(ConfigProperty('nlp', 'nlp.openai.api_key', 'YOUR-API-KEY'))
agent.add_property(ConfigProperty('nlp', 'nlp.hf.api_key', 'YOUR-API-KEY'))
agent.add_property(ConfigProperty('nlp', 'nlp.replicate.api_key', 'YOUR-API-KEY'))

# INTENTS
muscles_intent = agent.new_intent('Muscles_intent', [
    'I want muscles',
],
description="Question about how to become muscular and which exercises to perform.")
nutrition_intent = agent.new_intent('Nutrition_intent', [
    'Which food to eat?',
],
description="Question about nutrition in gym.")
other = agent.new_intent('Other', [
],
description="Any question that does not fit in \"Muscles\" or \"Nutrition\"")

# Create LLM instance for use in state bodies
llm = LLMOpenAI(agent=agent, name='gpt-4o-mini', parameters={})

# STATES
initial = agent.new_state('Initial', initial=True)
idle = agent.new_state('Idle')
trainingplan = agent.new_state('TrainingPlan')
nutrition = agent.new_state('Nutrition')
otherquestions = agent.new_state('OtherQuestions')

# Initial state
initial_body = Body('Initial_body')
initial_body.add_action(AgentReply('Hey, I’m your fitness advisor.'))

initial.set_body(initial_body)
initial.go_to(idle)

# Idle state
idle_body = Body('Idle_body')
idle_body.add_action(AgentReply('Got questions about workouts, food, or recovery? I’ve got you.'))

idle.set_body(idle_body)
idle_fallback_body = Body('Idle_fallback_body')
idle_fallback_body.add_action(AgentReply('I’m here just to talk gym and fitness.'))

idle.set_fallback_body(idle_fallback_body)
idle.when_intent_matched(muscles_intent).go_to(trainingplan)

idle.when_intent_matched(other).go_to(otherquestions)

idle.when_intent_matched(nutrition_intent).go_to(nutrition)

# TrainingPlan state
trainingplan_body = Body('TrainingPlan_body')
trainingplan_body.add_action(AgentReply('Focus on the big, heavy lifts—squats, deadlifts, bench, overhead press, and rows. That’s how you build muscle.'))
trainingplan_body.add_action(AgentReply('Train 3–5 days a week. Nudge the weight up over time. Eat enough protein and calories to grow.'))
trainingplan_body.add_action(AgentReply('Get enough sleep and stay consistent. Muscle grows from steady work over time.'))

trainingplan.set_body(trainingplan_body)
trainingplan.go_to(idle)

# Nutrition state
nutrition_body = Body('Nutrition_body')
nutrition_body.add_action(AgentReply('Food basics: mostly whole foods—lean protein, veggies, fruit, whole grains, and healthy fats. Aim for about 1.6–2.2 grams of protein per kilogram of body weight each day.'))
nutrition_body.add_action(AgentReply('Match your calories to your goal: a little more to gain muscle, a little less to lose fat, or the same to keep your weight.'))
nutrition_body.add_action(AgentReply('Carbs fuel your training. Fats support your hormones.'))
nutrition_body.add_action(AgentReply('Drink plenty of water. Go easy on ultra-processed foods and alcohol.'))
nutrition_body.add_action(AgentReply('Be consistent. Results come from steady habits, not perfection.'))

nutrition.set_body(nutrition_body)
nutrition.go_to(idle)

# OtherQuestions state
otherquestions_body = Body('OtherQuestions_body')
otherquestions_body.add_action(LLMReply())

otherquestions.set_body(otherquestions_body)
otherquestions.go_to(idle)

