import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/web";
import { HumanMessage, AIMessage, BaseMessage, SystemMessage } from "@langchain/core/messages";
import { LinearClient, LinearDocument as L } from "@linear/sdk";
import { getCoordinates, getWeather, getTime } from "./tools";
import { prompt } from "./prompt";
import { Content, isToolName, ToolName, UnreachableCaseError } from "../types";

export class AgentClient {
  private linearClient: LinearClient;
  private model: ChatAnthropic;
  private agent: any; // LangGraph agent

  // Maximum number of iterations for the agent to prevent infinite loops
  private MAX_ITERATIONS = 10;

  constructor(linearAccessToken: string, anthropicApiKey: string) {
    this.linearClient = new LinearClient({
      accessToken: linearAccessToken,
    });
    this.model = new ChatAnthropic({
      apiKey: anthropicApiKey,
      model: "claude-3-5-sonnet-latest",
      temperature: 0,
    });

    // Create the LangGraph agent with tools
    this.agent = createReactAgent({
      llm: this.model,
      tools: [getCoordinates, getWeather, getTime],
    });
  }

  /**
   * Handle a user prompt by processing it through the agent.
   * @param userPrompt - The user prompt
   * @param agentSessionId - The Linear agent session ID
   */
  public async handleUserPrompt(agentSessionId: string, userPrompt: string) {
    // Generate more context for the LLM from previous activities in this agent session
    const activities = await this.generateMessagesFromPreviousActivities(
      agentSessionId
    );

    const messages: BaseMessage[] = [
      new HumanMessage(userPrompt || ""),
      ...activities,
    ];

    let taskComplete = false;
    let iterations = 0;

    while (!taskComplete && iterations < this.MAX_ITERATIONS) {
      iterations++;

      try {
        const response = await this.callLangGraphAgent(messages);
        const content = this.mapResponseToLinearActivityContent(response);

        if (content.type === L.AgentActivityType.Thought) {
          await this.linearClient.createAgentActivity({
            agentSessionId,
            content,
          });

          // Add to conversation history
          messages.push(new AIMessage(response));

          // Continue the loop for next cycle, until the task is complete or the maximum iteration count is reached
          await new Promise((resolve) => setTimeout(resolve, 1000));
        } else if (content.type === L.AgentActivityType.Action) {
          const toolName = content.action;
          // PART 1: Create the action activity to inform the user that the agent is going to use the tool
          await this.linearClient.createAgentActivity({
            agentSessionId,
            content,
          });

          // PART 2: Execute the tool
          const parameter = content.parameter;
          const toolResult = await this.executeAction({
            action: toolName,
            parameter,
          });

          // Add tool result to conversation for next LLM call
          messages.push(new AIMessage(response));
          messages.push(new HumanMessage(`Tool result: ${toolResult}`));

          // PART 3: Create the result activity to inform the user that the tool has been executed
          const resultContent: Content = {
            type: L.AgentActivityType.Action,
            action: toolName,
            result: toolResult,
            parameter: parameter || null,
          };
          await this.linearClient.createAgentActivity({
            agentSessionId,
            content: resultContent,
          });

          // Continue the loop for next cycle
          await new Promise((resolve) => setTimeout(resolve, 1000));
        } else if (content.type === L.AgentActivityType.Response) {
          await this.linearClient.createAgentActivity({
            agentSessionId,
            content,
          });
          taskComplete = true;
        } else if (content.type === L.AgentActivityType.Error) {
          await this.linearClient.createAgentActivity({
            agentSessionId,
            content,
          });
          taskComplete = true;
        } else if (content.type === L.AgentActivityType.Elicitation) {
          await this.linearClient.createAgentActivity({
            agentSessionId,
            content,
          });
          taskComplete = true;
        }
      } catch (error) {
        const errorMessage = `Agent error: ${
          error instanceof Error ? error.message : "Unknown error"
        }`;
        await this.linearClient.createAgentActivity({
          agentSessionId,
          content: {
            type: "error",
            body: errorMessage,
          },
        });
        taskComplete = true;
      }
    }

    if (!taskComplete && iterations >= this.MAX_ITERATIONS) {
      const maxIterationsMessage =
        "The agent has reached the maximum number of iterations and will now stop.";
      await this.linearClient.createAgentActivity({
        agentSessionId,
        content: {
          type: "error",
          body: maxIterationsMessage,
        },
      });
    }
  }

  /**
   * Map the response from the OpenAI API to content for an agent activity in Linear.
   * In the case of an action, this function will return a different content object if the action has been executed.
   *
   * @param response - The response from the OpenAI API
   * @returns The Linear activity type
   */
  private mapResponseToLinearActivityContent(response: string): Content {
    const typeToKeyword = {
      [L.AgentActivityType.Thought]: "THINKING:",
      [L.AgentActivityType.Action]: "ACTION:",
      [L.AgentActivityType.Response]: "RESPONSE:",
      [L.AgentActivityType.Elicitation]: "ELICITATION:",
      [L.AgentActivityType.Error]: "ERROR:",
    } as const;
    const mappedType = Object.entries(typeToKeyword).find(([_, keyword]) =>
      response.startsWith(keyword)
    );
    const type = mappedType?.[0]
      ? (mappedType[0] as L.AgentActivityType)
      : L.AgentActivityType.Thought;

    switch (type) {
      case L.AgentActivityType.Thought:
      case L.AgentActivityType.Response:
      case L.AgentActivityType.Elicitation:
      case L.AgentActivityType.Error:
        return { type, body: response.replace(typeToKeyword[type], "").trim() };
      case L.AgentActivityType.Action:
        // Parse action parameters
        const actionMatch = response.match(/ACTION:\s*(\w+)\(([^)]+)\)/);
        if (actionMatch) {
          const [, toolNameRaw, params] = actionMatch;
          if (!isToolName(toolNameRaw)) {
            throw new Error(`Invalid tool name: ${toolNameRaw}`);
          }
          const toolName = toolNameRaw as ToolName;
          return {
            type,
            action: toolName,
            parameter: params || null,
          };
        }
      default:
        throw new UnreachableCaseError(type);
    }
  }

  /**
   * Execute an action and return the result
   * @param props - The action and parameter
   * @returns The result of the action
   */
  private async executeAction(props: {
    action: ToolName;
    parameter: string | null;
  }): Promise<string> {
    const { action, parameter } = props;
    if (!parameter) {
      throw new Error("Parameter is required for action execution");
    }
    switch (action) {
      case "getCoordinates":
        return JSON.stringify(
          await getCoordinates(parameter.replace(/"/g, ""))
        );
      case "getWeather":
        const paramParts = parameter
          .split(",")
          .map((p: string) => parseFloat(p.trim()));
        if (
          paramParts.length >= 2 &&
          !isNaN(paramParts[0]) &&
          !isNaN(paramParts[1])
        ) {
          const lat = paramParts[0]!;
          const long = paramParts[1]!;
          return JSON.stringify(await getWeather({ lat, long }));
        } else {
          throw new Error("Invalid parameter for getWeather action");
        }
      case "getTime":
        const timeParamParts = parameter
          .split(",")
          .map((p: string) => parseFloat(p.trim()));
        if (
          timeParamParts.length >= 2 &&
          !isNaN(timeParamParts[0]) &&
          !isNaN(timeParamParts[1])
        ) {
          const lat = timeParamParts[0]!;
          const long = timeParamParts[1]!;
          return await getTime({ lat, long });
        } else {
          throw new Error("Invalid parameter for getTime action");
        }
      default:
        throw new UnreachableCaseError(action);
    }
  }

  /**
   * Call the LangGraph agent to get a response
   * @param messages - The messages to send to the agent
   * @returns The response from the agent
   */
  private async callLangGraphAgent(messages: BaseMessage[]): Promise<string> {
    try {
      // Invoke the model directly with system prompt and messages
      const systemMessage = new SystemMessage(prompt);
      const allMessages = [systemMessage, ...messages];

      const response = await this.model.invoke(allMessages);

      return response.content as string;
    } catch (error) {
      throw new Error(
        `Anthropic API error: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  /**
   * Generate additional context for the LLM from previous activities in this agent session.
   * In our case, we only consider user prompt and agent responses, but you can extend this logic as needed.
   *
   * @param agentSessionId - The Linear agent session ID
   * @returns All activities for the agent session
   */
  private async generateMessagesFromPreviousActivities(
    agentSessionId: string
  ): Promise<BaseMessage[]> {
    const agentSession = await this.linearClient.agentSession(agentSessionId);

    // Get all activities with pagination
    const allActivities = [];
    let activitiesConnection = await agentSession.activities();
    let hasNextPage = activitiesConnection.pageInfo.hasNextPage;

    // Add first page of activities
    allActivities.push(...activitiesConnection.nodes);

    // Continue fetching while there are more pages
    while (hasNextPage && activitiesConnection.pageInfo.endCursor) {
      activitiesConnection = await agentSession.activities({
        after: activitiesConnection.pageInfo.endCursor,
      });
      allActivities.push(...activitiesConnection.nodes);
      hasNextPage = activitiesConnection.pageInfo.hasNextPage;
    }

    const activities: BaseMessage[] = [];
    for (const activity of allActivities
      .filter(
        (activity) =>
          activity.content.type === L.AgentActivityType.Prompt ||
          activity.content.type === L.AgentActivityType.Response
      )
      .reverse()) {
      const typedContent = activity.content as
        | L.AgentActivityPromptContent
        | L.AgentActivityResponseContent;
      const content = typedContent.body;

      if (activity.content.type === L.AgentActivityType.Prompt) {
        activities.push(new HumanMessage(content));
      } else {
        activities.push(new AIMessage(content));
      }
    }
    return activities;
  }
}
