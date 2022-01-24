package ifmo.mmmlpmsw.cloud.bot

import java.util.*


class CatBotProperties {

    companion object {
        private const val PROPERTIES_FILE = "/bot.properties"
        private val props: Properties = Properties()

        init {
            try {
                props.load(this::class.java.getResourceAsStream(PROPERTIES_FILE))
            } catch (e: NullPointerException) {
                System.err.println("Settings file '$PROPERTIES_FILE' not found")
            }
        }

        val botUsername: String = props.getProperty("bot_username")
        val botToken: String = props.getProperty("bot_token")
    }
}
