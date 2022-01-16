package ifmo.mmmlpmsw.cloud.bot

import java.io.FileInputStream
import java.util.Properties;

class CatBotProperties {
    companion object {
        private val PROPERTIES_FILE = "bot.properties"
        private var props: Properties = Properties()

        init {
            try {
                props.load(this::class.java.classLoader.getResourceAsStream(PROPERTIES_FILE))
            } catch (e: NullPointerException) {
                System.err.println("Settings file '$PROPERTIES_FILE' not found")
            }
        }

        val botUsername: String = props.getProperty("bot_username")

        val botToken: String = props.getProperty("bot_token")
    }

}