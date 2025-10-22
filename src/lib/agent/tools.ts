import { tool } from "@langchain/core/tools";
import { z } from "zod";

/**
 * Get the coordinates for a given city name (internal implementation)
 * @param city_name - The name of the city
 * @returns The coordinates for the given city
 */
const getCoordinatesImpl = async (
  city_name: string
): Promise<
  { lat: number; lon: number; displayName: string } | { error: string }
> => {
  try {
    const response = await fetch(
      `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(
        city_name
      )}&format=jsonv2`,
      {
        headers: {
          "User-Agent": "Linear-Demo-Agent/1.0 (https://demo-agent.linear.dev)",
          Accept: "application/json",
        },
      }
    );

    if (!response.ok) {
      return {
        error: `OpenStreetMap API error: ${response.status} ${response.statusText}`,
      };
    }

    const data = (await response.json()) as {
      lat: string;
      lon: string;
      display_name: string;
    }[];

    if (data && data.length > 0) {
      const result = data[0];
      return {
        lat: parseFloat(result.lat),
        lon: parseFloat(result.lon),
        displayName: result.display_name,
      };
    } else {
      return { error: "Location not found" };
    }
  } catch (error) {
    return {
      error: `Failed to get coordinates: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
    };
  }
};

/**
 * Get the weather for a given location (internal implementation)
 * @param lat - The latitude of the location
 * @param long - The longitude of the location
 * @returns The weather for the given location
 */
const getWeatherImpl = async (params: {
  lat: number;
  long: number;
}): Promise<string> => {
  const { lat, long } = params;
  try {
    const response = await fetch(
      `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${long}&current=temperature_2m,weathercode`
    );
    const data = (await response.json()) as {
      current: { temperature_2m: number; weathercode: number };
    };

    if (data && data.current) {
      const { temperature_2m, weathercode } = data.current;

      // Convert weather code to human readable description
      const getWeatherDescription = (code: number): string => {
        const weatherCodes: { [key: number]: string } = {
          0: "clear sky",
          1: "mainly clear",
          2: "partly cloudy",
          3: "overcast",
          45: "fog",
          48: "depositing rime fog",
          51: "light drizzle",
          53: "moderate drizzle",
          55: "dense drizzle",
          56: "light freezing drizzle",
          57: "dense freezing drizzle",
          61: "slight rain",
          63: "moderate rain",
          65: "heavy rain",
          66: "light freezing rain",
          67: "heavy freezing rain",
          71: "slight snow",
          73: "moderate snow",
          75: "heavy snow",
          77: "snow grains",
          80: "slight rain showers",
          81: "moderate rain showers",
          82: "violent rain showers",
          85: "slight snow showers",
          86: "heavy snow showers",
          95: "thunderstorm",
          96: "thunderstorm with slight hail",
          99: "thunderstorm with heavy hail",
        };
        return weatherCodes[code] || "unknown weather";
      };

      const weatherDescription = getWeatherDescription(weathercode);
      return `${temperature_2m}°C, ${weatherDescription}`;
    } else {
      console.log("Weather data not available", data);
      return "Weather data not available";
    }
  } catch (error) {
    console.log("Error getting weather", error);
    return `Failed to get weather: ${
      error instanceof Error ? error.message : "Unknown error"
    }`;
  }
};

/**
 * Get the current time for a given location (internal implementation)
 * @param lat - The latitude of the location
 * @param long - The longitude of the location
 * @returns The current time for the given location
 */
const getTimeImpl = async (params: {
  lat: number;
  long: number;
}): Promise<string> => {
  const { lat, long } = params;
  try {
    const controller = new AbortController();
    // 60 second timeout because this free time API is very slow!
    const timeoutId = setTimeout(() => controller.abort(), 60000);

    const response = await fetch(
      `https://timeapi.io/api/Time/current/coordinate?latitude=${lat}&longitude=${long}`,
      {
        signal: controller.signal,
      }
    );

    clearTimeout(timeoutId);

    if (!response.ok) {
      return `Time API error: ${response.status} ${response.statusText}`;
    }

    const data = (await response.json()) as {
      date: string;
      time: string;
      timeZone: string;
      dayOfWeek: string;
      dstActive: boolean;
    };

    if (data) {
      const { date, time, timeZone, dayOfWeek, dstActive } = data;
      const dstStatus = dstActive ? " (DST active)" : "";
      return `${dayOfWeek}, ${date} at ${time} ${timeZone}${dstStatus}`;
    } else {
      console.log("Time data not available", data);
      return "Time data not available";
    }
  } catch (error) {
    console.log("Error getting time", error);
    return `Failed to get time: ${
      error instanceof Error ? error.message : "Unknown error"
    }`;
  }
};

// Export LangChain tool wrappers for use with LangGraph
export const getCoordinates = tool(
  async ({ city_name }) => {
    const result = await getCoordinatesImpl(city_name);
    return JSON.stringify(result);
  },
  {
    name: "getCoordinates",
    description: "Get coordinates (latitude and longitude) for a given city name",
    schema: z.object({
      city_name: z.string().describe("The name of the city to get coordinates for"),
    }),
  }
);

export const getWeather = tool(
  async ({ lat, long }) => {
    const result = await getWeatherImpl({ lat, long });
    return result;
  },
  {
    name: "getWeather",
    description: "Get current weather for given coordinates (latitude first, then longitude)",
    schema: z.object({
      lat: z.number().describe("The latitude of the location"),
      long: z.number().describe("The longitude of the location"),
    }),
  }
);

export const getTime = tool(
  async ({ lat, long }) => {
    const result = await getTimeImpl({ lat, long });
    return result;
  },
  {
    name: "getTime",
    description: "Get current time for given coordinates (latitude first, then longitude)",
    schema: z.object({
      lat: z.number().describe("The latitude of the location"),
      long: z.number().describe("The longitude of the location"),
    }),
  }
);
