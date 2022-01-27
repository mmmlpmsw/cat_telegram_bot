package ifmo.mmmlpmsw.cloud.server

import org.springframework.core.io.Resource
import org.springframework.stereotype.Service
import org.springframework.web.client.RestTemplate


@Service
class CatsService {

    companion object {

        private const val SERVER_URL = "https://cats-python-pictures.azurewebsites.net/?emotion={emotion}"
    }

    fun getCatForEmotion(emotion: Emotion): Resource? = try {
        RestTemplate().getForObject(SERVER_URL, Resource::class.java, mapOf("emotion" to emotion.getName()))
    } catch (e: Throwable) {
        null
    }
}
