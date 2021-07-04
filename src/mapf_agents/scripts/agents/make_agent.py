from agents.user_agent import UserAgent

def make_agent(agent_type):
	if agent_type == 'user':
		return UserAgent()

	raise NotImplementedError()
