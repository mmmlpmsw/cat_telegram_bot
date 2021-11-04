package ifmo.mmmlpmsw.cloud.server

import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class Controller {
    @GetMapping("/", produces = [MediaType.IMAGE_PNG_VALUE])
    fun getPictureWithCat(): ByteArray {
        return javaClass.classLoader.getResourceAsStream("0.png")?.readAllBytes()!!
    }
}