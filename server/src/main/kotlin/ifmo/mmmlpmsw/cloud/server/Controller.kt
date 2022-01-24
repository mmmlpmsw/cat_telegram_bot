package ifmo.mmmlpmsw.cloud.server

import org.springframework.beans.factory.annotation.Value
import org.springframework.core.io.Resource
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController


@RestController
class Controller(
    private val emotionsService: EmotionsService,
    private val catsService: CatsService,
    @Value("classpath:0.png")
    private val defaultCat: Resource,
    @Value("classpath:1.jpg")
    private val defaultCat1: Resource,
    @Value("classpath:cat.jpeg")
    private val defaultCat2: Resource,
) {

    private val defaultCats = listOf(defaultCat, defaultCat1, defaultCat2)

    @GetMapping("/", produces = [MediaType.IMAGE_PNG_VALUE])
    fun getPictureWithCat(@RequestParam text: String): Resource =
        getCatByEmotion(getEmotionOfText(text))

    @GetMapping("/emotion")
    fun getEmotionOfText(@RequestParam text: String): String? =
        emotionsService.getEmotionOfText(text)?.getName()

    @GetMapping("/cat", produces = [MediaType.IMAGE_PNG_VALUE])
    fun getCatByEmotion(@RequestParam(required = false) emotion: String?): Resource =
        emotion?.let(Emotion::getByName)?.let(catsService::getCatForEmotion)
            ?: pickDefaultCat()

    private fun pickDefaultCat() = defaultCats.random()
}
