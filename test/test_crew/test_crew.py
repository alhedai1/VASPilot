from vaspilot.crew.vasp_crew import VaspCrew
from crewai import Task
import yaml


config = yaml.load(open("/data/work/jxliu/LLM/Agent/llm_research/MCP/vasp_crew/configs/config.yaml"), Loader=yaml.FullLoader)
generator = VaspCrew(config)
crew = generator.crew()
task_1 = "用ENMAX=300~500每50eV取一个点，做scf计算MoS2的总能，并画出总能与enmax的关系，不需要做结构弛豫结构文件位于/data/work/jxliu/LLM/Agent/llm_research/MCP/crew_AI_agent/POSCAR。"
task_2 = "用ENMAX=500计算MoS2沿GMKG路径的能带，并画出能带图，不需要做结构弛豫。结构文件位于/data/work/jxliu/LLM/Agent/llm_research/MCP/crew_AI_agent/POSCAR。"
task_4 = "用ENMAX=300, 400, 500计算MoS2沿GMKG路径的能带，并将能带图画在一张图上。结构文件位于/data/work/jxliu/LLM/Agent/llm_research/MCP/crew_AI_agent/POSCAR。"
task_5 = "用ENCUT=500 分别对AD.vasp(2层), ADA.vasp(3层), ADAD.vasp(4层)和ADADA.vasp(5层)进行弛豫和自洽计算。弛豫时需要指定ISIF=4和IVDW=11。这些结构文件位于/data/work/jxliu/Hetero_Struct/2H-MoS2/multi_layer/scripts/structures/目录下。计算完成后，将带隙的变化画出来。"
task_6 = "下载2H-MoS2的结构文件，用IVDW=11做结构弛豫、并画出GMKG路径上的能带图。"

calc_task = Task(
        description=task_1,
        expected_output="一份详尽的报告，报告内容包括任务执行过程、画出的图表位置。",
        output_file='calculate_report.md',
        human_input=True,
    )
crew.tasks = [calc_task]
crew.kickoff()