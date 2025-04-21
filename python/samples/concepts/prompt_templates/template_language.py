# Copyright (c) Microsoft. All rights reserved.

import asyncio


from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.core_plugins import TimePlugin
from semantic_kernel.prompt_template import KernelPromptTemplate, PromptTemplateConfig


async def main():
    kernel = Kernel()

    # Add the Azure OpenAI chat completion service
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    import os
    from dotenv import load_dotenv
    load_dotenv()
    # Setting up the deployment name
    deployment_name = os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME']
    # The API key for your Azure OpenAI resource.
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    # The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
    azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']

    

    service_id = "template_language"
    kernel.add_service(AzureChatCompletion(service_id="template_language",
                                           deployment_name=deployment_name, 
                                           endpoint=azure_endpoint, 
                                           api_key=api_key))
    """ kernel.add_service(
        OpenAIChatCompletion(service_id=service_id), 
    ) """

    kernel.add_plugin(TimePlugin(), "time")

    function_definition = """
    Today is: {{time.date}}
    Current time is: {{time.time}}

    Answer to the following questions using JSON syntax, including the data used.
    Is it morning, afternoon, evening, or night (morning/afternoon/evening/night)?
    Is it weekend time (weekend/not weekend)?
    """

    print("--- Rendered Prompt ---")
    prompt_template_config = PromptTemplateConfig(template=function_definition)
    prompt_template = KernelPromptTemplate(prompt_template_config=prompt_template_config)
    rendered_prompt = await prompt_template.render(kernel, arguments=None)
    print(rendered_prompt)

    kind_of_day = kernel.add_function(
        plugin_name="TimePlugin",
        template=function_definition,
        execution_settings=OpenAIChatPromptExecutionSettings(service_id=service_id, max_tokens=100),
        function_name="kind_of_day",
        prompt_template=prompt_template,
    )

    print("--- Prompt Function Result ---")
    result = await kernel.invoke(function=kind_of_day)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
