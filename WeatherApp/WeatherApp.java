import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import org.json.JSONException;
import org.json.JSONObject;

public class WeatherApp {

    public static void main(String[] args) {
        String city = "London";
        String apiKey = "0ebb8d189715c75ad5761cf088e67f15";

        try {
            String weatherData = fetchWeatherData(city, apiKey);
            double temperature = extractTemperature(weatherData);

            System.out.println("Temperature in " + city + ": " + temperature + "°F");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String fetchWeatherData(String city, String apiKey) throws IOException {
        String apiUrl = "http://api.openweathermap.org/data/2.5/weather?q=" + city + "&APPID=" + apiKey;

        URL url = new URL(apiUrl);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");

        BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
        StringBuilder response = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            response.append(line);
        }
        reader.close();

        return response.toString();
    }

    private static double extractTemperature(String weatherData) {
        try {
            JSONObject json = new JSONObject(weatherData);
            System.out.println(json);
            JSONObject main = json.getJSONObject("main");
            double temperatureK = main.getDouble("temp");
    
            double temperatureF = (temperatureK - 273.15) * 9/5 + 32;

            return temperatureF;
        } catch (JSONException e) {
            e.printStackTrace();
            return 0.0; // Return a default value or handle the error case as needed
        }
    }
}