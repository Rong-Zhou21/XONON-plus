package com.minerl.multiagent.env;

import com.microsoft.Malmo.Schemas.MissionInit;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import java.io.StringReader;

public class MissionSpec {
    public static MissionInit decodeMissionInit(String command)
    {
        try {
            JAXBContext context = JAXBContext.newInstance(MissionInit.class);
            return (MissionInit) context.createUnmarshaller()
                    .unmarshal(new StringReader(command));
        } catch (JAXBException e) {
            System.out.println("JAXB exception: " + e);
            throw new RuntimeException(e);
        }
    }
}
