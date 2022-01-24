package ifmo.mmmlpmsw.cloud.server

import org.jetbrains.kotlin.util.capitalizeDecapitalize.toLowerCaseAsciiOnly
import org.jetbrains.kotlin.util.capitalizeDecapitalize.toUpperCaseAsciiOnly


enum class Emotion {

    WORRY,
    ANGER,
    HATE,
    EMPTY,
    NEUTRAL,
    RELIEF,
    LOVE,
    HAPPINESS,
    FUN,
    SURPRISE,
    ENTHUSIASM,
    SADNESS,
    BOREDOM;


    fun getName() = name.toLowerCaseAsciiOnly()

    companion object {

        fun getByName(emotion: String) = try {
            valueOf(emotion.toUpperCaseAsciiOnly())
        } catch (e: IllegalArgumentException) {
            null
        }
    }
}
