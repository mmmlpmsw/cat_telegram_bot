package ifmo.mmmlpmsw.cloud.server

import org.springframework.stereotype.Service
import org.springframework.web.client.RestTemplate


@Service
class EmotionsService {

    companion object {

        private const val SERVER_URL = "https://emotions-server.azurewebsites.net/?text={text}"
    }

    fun getEmotionOfText(text: String): Emotion? = try {
        RestTemplate().getForObject(SERVER_URL, String::class.java, mapOf("text" to text))
            ?.let(Emotion::getByName)
    } catch (e: Throwable) {
        null
    }
}
