package ifmo.mmmlpmsw.cloud.server

import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@RestController
class Controller {
    @GetMapping("/", produces = [MediaType.IMAGE_JPEG_VALUE])
    fun getPictureWithCat(@RequestParam file: String): ResponseEntity<ByteArray> {
        val stream = javaClass.classLoader.getResourceAsStream(file)
        return if (stream == null)
            ResponseEntity.notFound().build()
        else
            ResponseEntity.ok(stream.readAllBytes())
    }
}