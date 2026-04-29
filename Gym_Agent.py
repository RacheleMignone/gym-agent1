# You may need to add your working directory to the Python path. To do so, uncomment the following lines of code
# import sys
# sys.path.append("/Path/to/directory/besser-agentic-framework") # Replace with your directory path

import json
import logging
import operator

from baf.core.agent import Agent
from baf.library.transition.events.base_events import *
from baf.nlp.llm.llm_huggingface import LLMHuggingFace
from baf.nlp.llm.llm_huggingface_api import LLMHuggingFaceAPI
from baf.nlp.llm.llm_openai_api import LLMOpenAI
from baf.nlp.llm.llm_replicate_api import LLMReplicate
from baf.core.session import Session
from baf.nlp.intent_classifier.intent_classifier_configuration import LLMIntentClassifierConfiguration, SimpleIntentClassifierConfiguration
from baf.nlp.speech2text.openai_speech2text import OpenAISpeech2Text
from baf.nlp.text2speech.openai_text2speech import OpenAIText2Speech

# Configure the logging module
logging.basicConfig(level=logging.INFO, format='{levelname} - {asctime}: {message}', style='{')

def build_agent_configuration_prompt(agent_configuration: dict) -> str:
    base_prompt = """
You are a personalized AI assistant.

You will be provided with an object called `agent_configuration` as a JSON string.
This object defines how you should present yourself and formulate your responses.

It controls stylistic and presentation aspects such as:
- Response language
- Level of formality
- Language complexity
- Sentence length
- And more

You must strictly follow the preferences defined in `agent_configuration` for all responses.

Here is the current agent configuration (JSON):

{agent_configuration_json}
"""
    return base_prompt.format(agent_configuration_json=json.dumps(agent_configuration, indent=2))


def build_user_profile_prompt(user_profile: dict) -> str:
    profile_prompt = """
You have access to the current user's profile encoded as JSON.
Leverage these traits whenever the agent configuration instructs you to adapt content.

Here is the active user profile (JSON):

{user_profile_json}
"""
    return profile_prompt.format(user_profile_json=json.dumps(user_profile, indent=2))


# Create the bot
agent = Agent('Gym_Agent', user_profiles_path='user_profiles.json', persist_sessions=True)
# Load bot properties stored in a dedicated file
agent.load_properties('config.yaml')

# Define the platform your chatbot will use

# Collect profile names from provided personalization mappings
profile_names = []
agent_configurations = {}
user_profiles = {}
context_prompts = {}
profile_prompts = {}
profile_names.append('User 2')
agent_configurations['User_2'] = json.loads(r'''{
  "adaptContentToUserProfile": true,
  "agentLanguage": "original",
  "agentPlatform": "streamlit",
  "agentStyle": "informal",
  "avatar": null,
  "inputModalities": [
    "text",
    "speech"
  ],
  "intentRecognitionTechnology": "llm-based",
  "interfaceStyle": {
    "alignment": "left",
    "color": "var(--apollon-primary-contrast)",
    "contrast": "medium",
    "font": "sans",
    "lineSpacing": 2,
    "size": 18
  },
  "languageComplexity": "original",
  "llm": {
    "model": "gpt-5",
    "provider": "openai"
  },
  "outputModalities": [
    "text",
    "speech"
  ],
  "responseTiming": "instant",
  "sentenceLength": "original",
  "useAbbreviations": false,
  "userProfileName": "User 2",
  "voiceStyle": {
    "gender": "male",
    "speed": 1
  }
}''')
user_profiles['User_2'] = json.loads(r'''{
  "model": {
    "Accessibility": {
      "Disability": {
        "affects": "Mobility",
        "description": "can\u0027t use lower body",
        "name": "Paraplegic"
      }
    },
    "class": "User",
    "id": "user_1"
  },
  "name": "UserProfile"
}''')
context_prompts['User_2'] = build_agent_configuration_prompt(agent_configurations['User_2'])
profile_prompts['User_2'] = build_user_profile_prompt(user_profiles['User_2'])
profile_names.append('User Diagram')
agent_configurations['User_Diagram'] = json.loads(r'''{
  "adaptContentToUserProfile": true,
  "agentLanguage": "original",
  "agentPlatform": "streamlit",
  "agentStyle": "informal",
  "avatar": null,
  "inputModalities": [
    "text",
    "speech"
  ],
  "intentRecognitionTechnology": "llm-based",
  "interfaceStyle": {
    "alignment": "left",
    "color": "var(--apollon-primary-contrast)",
    "contrast": "medium",
    "font": "sans",
    "lineSpacing": 2,
    "size": 18
  },
  "languageComplexity": "original",
  "llm": {
    "model": "gpt-5",
    "provider": "openai"
  },
  "outputModalities": [
    "text",
    "speech"
  ],
  "responseTiming": "instant",
  "sentenceLength": "original",
  "useAbbreviations": false,
  "userProfileName": "User 2",
  "voiceStyle": {
    "gender": "male",
    "speed": 1
  }
}''')
user_profiles['User_Diagram'] = json.loads(r'''{
  "model": {
    "Personal_Information": {
      "age": 65
    },
    "class": "User",
    "id": "user_1"
  },
  "name": "UserProfile"
}''')
context_prompts['User_Diagram'] = build_agent_configuration_prompt(agent_configurations['User_Diagram'])
agent.set_agent_configurations(agent_configurations)




platform = agent.use_websocket_platform(use_ui=True, authenticate_users=True)
# LLM instantiation based on config['llm']
reply_llm = LLMOpenAI(
    agent=agent,
    name='gpt-5',
    parameters={}
)

stt = OpenAISpeech2Text(agent=agent, model_name="whisper-1")
tts = OpenAIText2Speech(agent=agent, model_name="gpt-4o-mini-tts")




ic_config = LLMIntentClassifierConfiguration(
    llm_name='gpt-5',
    parameters={},
    use_intent_descriptions=True,
    use_training_sentences=True,
    use_entity_descriptions=False,
    use_entity_synonyms=False
)

agent.set_default_ic_config(ic_config)



##############################
# INTENTS
##############################
Muscles_intent = agent.new_intent('Muscles_intent', [
    'I want muscles',
    ],
    description='Question about how to become muscular and which exercises to perform.'
)
Nutrition_intent = agent.new_intent('Nutrition_intent', [
    'Which food to eat?',
    ],
    description='Question about nutrition in gym.'
)
Other = agent.new_intent('Other', [
    ],
    description='Any question that does not fit in "Muscles" or "Nutrition"'
)

##############################
# PERSONALIZED INTENTS
##############################
# Intents for profile User_2
Muscles_intent_User_2 = agent.new_intent('Muscles_intent_User_2', [
    'I want muscles',
    ],
    description='Question about how to become muscular and which exercises to perform.'
)
Nutrition_intent_User_2 = agent.new_intent('Nutrition_intent_User_2', [
    'Which food to eat?',
    ],
    description='Question about nutrition in gym.'
)
Other_User_2 = agent.new_intent('Other_User_2', [
    ],
    description='Any question that does not fit in "Muscles" or "Nutrition"'
)

# Intents for profile User_Diagram
Muscles_intent_User_Diagram = agent.new_intent('Muscles_intent_User_Diagram', [
    'I want muscles',
    ],
    description='Question about how to become muscular and which exercises to perform.'
)
Nutrition_intent_User_Diagram = agent.new_intent('Nutrition_intent_User_Diagram', [
    'Which food to eat?',
    ],
    description='Question about nutrition in gym.'
)
Other_User_Diagram = agent.new_intent('Other_User_Diagram', [
    ],
    description='Any question that does not fit in "Muscles" or "Nutrition"'
)



##############################
# CUSTOM CONDITIONS
##############################




##############################
# STATES
##############################

# Dummy entry state to fan out to profile-specific initial states
router_initial_state = agent.new_state('router_initial_state', initial=True)

Initial = agent.new_state('Initial')
Idle = agent.new_state('Idle')
TrainingPlan = agent.new_state('TrainingPlan')
Nutrition = agent.new_state('Nutrition')
OtherQuestions = agent.new_state('OtherQuestions')

##############################
# PROFILE STATES
##############################
# States for profile User_2
Initial_User_2 = agent.new_state('Initial_User_2')
Idle_User_2 = agent.new_state('Idle_User_2')
TrainingPlan_User_2 = agent.new_state('TrainingPlan_User_2')
Nutrition_User_2 = agent.new_state('Nutrition_User_2')
OtherQuestions_User_2 = agent.new_state('OtherQuestions_User_2')
# States for profile User_Diagram
Initial_User_Diagram = agent.new_state('Initial_User_Diagram')
Idle_User_Diagram = agent.new_state('Idle_User_Diagram')
TrainingPlan_User_Diagram = agent.new_state('TrainingPlan_User_Diagram')
Nutrition_User_Diagram = agent.new_state('Nutrition_User_Diagram')
OtherQuestions_User_Diagram = agent.new_state('OtherQuestions_User_Diagram')

##############################
# ROUTER TRANSITIONS TO PROFILE INITIAL STATES
##############################
router_initial_state.when_variable_matches_operation('user_profile', operator.eq, 'User 2').go_to(Initial_User_2)
router_initial_state.when_variable_matches_operation('user_profile', operator.eq, 'User Diagram').go_to(Initial_User_Diagram)


# Initial
def Initial_body(session: Session):
    speech_messages = []
    reply_text = 'Hey, I’m your fitness advisor.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Initial.set_body(Initial_body)
Initial.go_to(Idle)
# Idle
def Idle_body(session: Session):
    speech_messages = []
    reply_text = 'Got questions about workouts, food, or recovery? I’ve got you.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Idle.set_body(Idle_body)
def Idle_fallback_body(session: Session):
    speech_messages = []
    reply_text = 'I’m here just to talk gym and fitness.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Idle.set_fallback_body(Idle_fallback_body)
Idle.when_intent_matched(Muscles_intent).go_to(TrainingPlan)
Idle.when_intent_matched(Other).go_to(OtherQuestions)
Idle.when_intent_matched(Nutrition_intent).go_to(Nutrition)
Idle.when_variable_matches_operation('user_profile', operator.eq, 'User 2').go_to(router_initial_state)
Idle.when_variable_matches_operation('user_profile', operator.eq, 'User Diagram').go_to(router_initial_state)
# TrainingPlan
def TrainingPlan_body(session: Session):
    speech_messages = []
    reply_text = 'Focus on the big, heavy lifts—squats, deadlifts, bench, overhead press, and rows. That’s how you build muscle.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Train 3–5 days a week. Nudge the weight up over time. Eat enough protein and calories to grow.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Get enough sleep and stay consistent. Muscle grows from steady work over time.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
TrainingPlan.set_body(TrainingPlan_body)
TrainingPlan.go_to(Idle)
# Nutrition
def Nutrition_body(session: Session):
    speech_messages = []
    reply_text = 'Food basics: mostly whole foods—lean protein, veggies, fruit, whole grains, and healthy fats. Aim for about 1.6–2.2 grams of protein per kilogram of body weight each day.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Match your calories to your goal: a little more to gain muscle, a little less to lose fat, or the same to keep your weight.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Carbs fuel your training. Fats support your hormones.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Drink plenty of water. Go easy on ultra-processed foods and alcohol.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Be consistent. Results come from steady habits, not perfection.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Nutrition.set_body(Nutrition_body)
Nutrition.go_to(Idle)
# OtherQuestions
def OtherQuestions_body(session: Session):
    message = reply_llm.predict(message=session.event.message, session=session)
    session.reply(message)
    platform.reply_speech(session, message)
OtherQuestions.set_body(OtherQuestions_body)
OtherQuestions.go_to(Idle)



##############################
# PROFILE STATE BODIES & TRANSITIONS
##############################
# Initial (User_2)
def Initial_body_User_2(session: Session):
    reply_llm.add_user_context(
        session=session,
        context=context_prompts.get('User_2'),
        context_name='agent_configuration_User_2'
    )
    reply_llm.add_user_context(
        session=session,
        context=profile_prompts.get('User_2'),
        context_name='user_profile_User_2'
    )
    speech_messages = []
    reply_text = 'Hey, I’m your buddy—your go-to fitness agent.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Initial_User_2.set_body(Initial_body_User_2)
Initial_User_2.go_to(Idle_User_2)
# Idle (User_2)
def Idle_body_User_2(session: Session):
    speech_messages = []
    reply_text = 'I’m here to answer any questions about workouts, nutrition, and recovery.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Idle_User_2.set_body(Idle_body_User_2)
def Idle_fallback_body_User_2(session: Session):
    speech_messages = []
    reply_text = 'I stick to gym and fitness stuff only, dude.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Idle_User_2.set_fallback_body(Idle_fallback_body_User_2)
Idle_User_2.when_intent_matched(Muscles_intent_User_2).go_to(TrainingPlan_User_2)
Idle_User_2.when_intent_matched(Other_User_2).go_to(OtherQuestions_User_2)
Idle_User_2.when_intent_matched(Nutrition_intent_User_2).go_to(Nutrition_User_2)
Idle_User_2.when_variable_matches_operation('user_profile', operator.eq, 'User Diagram').go_to(router_initial_state)
# TrainingPlan (User_2)
def TrainingPlan_body_User_2(session: Session):
    speech_messages = []
    reply_text = 'Stick with heavy compound lifts you can do from a seated or supported position—bench press, seated overhead press, seated or chest-supported rows, lat pulldowns or assisted pull-ups, and appropriate cable core work—to build overall muscle.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Train 3–5 times a week, keep bumping the weight up, and eat enough protein and calories to support growth.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Get solid sleep and stay consistent—muscle comes from steady effort over time.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
TrainingPlan_User_2.set_body(TrainingPlan_body_User_2)
TrainingPlan_User_2.go_to(Idle_User_2)
# Nutrition (User_2)
def Nutrition_body_User_2(session: Session):
    speech_messages = []
    reply_text = 'Nutrition basics: eat mostly whole foods (lean protein, veggies, fruit, whole grains, healthy fats). Protein: about 1.6–2.2 g per kg of body weight daily.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Match your calories to your goal: a small surplus to gain muscle, a deficit to lose fat, maintenance to stay the same.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Carbs fuel your training; fats support your hormones.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Drink plenty of water, and go easy on ultra-processed foods and alcohol.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Be consistent—results come from habits, not perfection.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Nutrition_User_2.set_body(Nutrition_body_User_2)
Nutrition_User_2.go_to(Idle_User_2)
# OtherQuestions (User_2)
def OtherQuestions_body_User_2(session: Session):
    message = reply_llm.predict(message=session.event.message, session=session)
    session.reply(message)
    platform.reply_speech(session, message)
OtherQuestions_User_2.set_body(OtherQuestions_body_User_2)
OtherQuestions_User_2.go_to(Idle_User_2)

# Initial (User_Diagram)
def Initial_body_User_Diagram(session: Session):
    reply_llm.add_user_context(
        session=session,
        context=context_prompts.get('User_Diagram'),
        context_name='agent_configuration_User_Diagram'
    )
    reply_llm.add_user_context(
        session=session,
        context=profile_prompts.get('User_Diagram'),
        context_name='user_profile_User_Diagram'
    )
    speech_messages = []
    reply_text = 'Hey, I’m your buddy—your go-to fitness agent.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Initial_User_Diagram.set_body(Initial_body_User_Diagram)
Initial_User_Diagram.go_to(Idle_User_Diagram)
# Idle (User_Diagram)
def Idle_body_User_Diagram(session: Session):
    speech_messages = []
    reply_text = 'I’m here to answer any questions about workouts, nutrition, and recovery.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Idle_User_Diagram.set_body(Idle_body_User_Diagram)
def Idle_fallback_body_User_Diagram(session: Session):
    speech_messages = []
    reply_text = 'I stick to gym and fitness stuff only, dude.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Idle_User_Diagram.set_fallback_body(Idle_fallback_body_User_Diagram)
Idle_User_Diagram.when_intent_matched(Muscles_intent_User_Diagram).go_to(TrainingPlan_User_Diagram)
Idle_User_Diagram.when_intent_matched(Other_User_Diagram).go_to(OtherQuestions_User_Diagram)
Idle_User_Diagram.when_intent_matched(Nutrition_intent_User_Diagram).go_to(Nutrition_User_Diagram)
Idle_User_Diagram.when_variable_matches_operation('user_profile', operator.eq, 'User 2').go_to(router_initial_state)
# TrainingPlan (User_Diagram)
def TrainingPlan_body_User_Diagram(session: Session):
    speech_messages = []
    reply_text = 'Stick with heavy compound lifts you can do from a seated or supported position—bench press, seated overhead press, seated or chest-supported rows, lat pulldowns or assisted pull-ups, and appropriate cable core work—to build overall muscle.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Train 3–5 times a week, keep bumping the weight up, and eat enough protein and calories to support growth.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Get solid sleep and stay consistent—muscle comes from steady effort over time.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
TrainingPlan_User_Diagram.set_body(TrainingPlan_body_User_Diagram)
TrainingPlan_User_Diagram.go_to(Idle_User_Diagram)
# Nutrition (User_Diagram)
def Nutrition_body_User_Diagram(session: Session):
    speech_messages = []
    reply_text = 'Nutrition basics: eat mostly whole foods (lean protein, veggies, fruit, whole grains, healthy fats). Protein: about 1.6–2.2 g per kg of body weight daily.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Match your calories to your goal: a small surplus to gain muscle, a deficit to lose fat, maintenance to stay the same.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Carbs fuel your training; fats support your hormones.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Drink plenty of water, and go easy on ultra-processed foods and alcohol.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    reply_text = 'Be consistent—results come from habits, not perfection.'
    session.reply(reply_text)
    speech_messages.append(reply_text)
    if speech_messages:
        platform.reply_speech(session, ' '.join(speech_messages))
Nutrition_User_Diagram.set_body(Nutrition_body_User_Diagram)
Nutrition_User_Diagram.go_to(Idle_User_Diagram)
# OtherQuestions (User_Diagram)
def OtherQuestions_body_User_Diagram(session: Session):
    message = reply_llm.predict(message=session.event.message, session=session)
    session.reply(message)
    platform.reply_speech(session, message)
OtherQuestions_User_Diagram.set_body(OtherQuestions_body_User_Diagram)
OtherQuestions_User_Diagram.go_to(Idle_User_Diagram)


# RUN APPLICATION

if __name__ == '__main__':
    agent.run()